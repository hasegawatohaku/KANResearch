import torch
import torch.nn as nn
import torch.nn.functional as F

from common.bspaline import make_knots, bspline_basis
from common.quantize import quantization_params, quantization_grid, quantize_values, quantize_indices

class KANELEEdgeLayer(nn.Module):
    """
    Common part of the quantized and pruned KA / KAonv layers with
    integer lookup tables.

    n_in = P inputs per output (patch elements for KAonv),
    n_out = O outputs.

    Parameters v [O, S] and u [O, P] have the same layout, the same
    initialization and the same creation order as in SaKANLayer and
    SaKANConv2dLayer. Grad-Free Spline: the spline basis is evaluated once
    at the K input levels and is a constant buffer.

    pre_silu=True applies SiLU to the layer input before quantization
    (the SiLU that SaKANPixelBlock places between two KA layers). It is
    folded into the table, because the input is the quantized output
    of the previous layer (same quantization range).
    """

    def __init__(
        self,
        n_in,
        n_out,
        grid_size,
        spline_order,
        bits,
        in_range,
        out_range,
        table_frac_bits,
        pre_silu=False
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bits = bits
        self.levels = 1 << bits
        self.in_range = (float(in_range[0]), float(in_range[1]))
        self.out_range = (float(out_range[0]), float(out_range[1]))
        self.frac_bits = int(table_frac_bits)
        self.pre_silu = pre_silu
        self.spline_order = spline_order
        self.n_spline = grid_size + spline_order

        self.out_scale, self.out_zero_point = quantization_params(
            bits,
            *self.out_range
        )

        # Integer output levels n = k - zero_point and table entry limits.
        self.n_min = -self.out_zero_point
        self.n_max = self.levels - 1 - self.out_zero_point
        self.entry_min = self.n_min * (1 << self.frac_bits)
        self.entry_max = self.n_max * (1 << self.frac_bits)

        # Sums of integer-valued float32 entries are exact below 2**24.
        if n_in * max(-self.entry_min, self.entry_max) >= 2 ** 24:
            raise ValueError(
                f"Integer table sums of a layer with {n_in} inputs exceed "
                f"2**24; reduce KANELE_TABLE_FRAC_BITS or KANELE_BITS."
            )

        self.v = nn.Parameter(
            0.02 * torch.randn(
                n_out,
                self.n_spline
            )
        )

        self.u = nn.Parameter(
            torch.empty(
                n_out,
                n_in
            )
        )

        nn.init.xavier_uniform_(self.u)

        self.register_buffer(
            "knots",
            make_knots(
                grid_size,
                spline_order
            )
        )

        self.register_buffer(
            "connection_mask",
            torch.ones(
                n_out,
                n_in
            )
        )

        # Input value of every LUT address k (SiLU folded in for pre_silu).
        points = quantization_grid(bits, *self.in_range)

        if pre_silu:
            points = quantize_values(F.silu(points), bits, *self.in_range)

        # [K, S], [K], [P]
        self.register_buffer(
            "basis_points",
            bspline_basis(points, self.knots, spline_order)
        )
        self.register_buffer("silu_points", F.silu(points))
        self.register_buffer("offsets", torch.arange(n_in) * self.levels)

        # Address histogram for pruning; collected only during calibration.
        self.pruning_stats = None

    # --------------------------------------------------------
    # Lookup tables
    # --------------------------------------------------------

    def table_values(self):
        """Connection outputs at all LUT addresses, value domain: [O, P, K]."""

        # [K, S] @ [S, O] -> [K, O]
        spline = self.basis_points @ self.v.t()

        table = (
            spline.t().unsqueeze(1) / self.n_in
            + self.u.unsqueeze(-1) * self.silu_points.view(1, 1, -1)
        )

        return table * self.connection_mask.unsqueeze(-1)

    def table_entries(self):
        """
        Integer table entries E[o,p,k] in units of out_scale / 2**F,
        fake-quantized with a clipped STE: [O, P, K].
        """

        unit = self.out_scale / (1 << self.frac_bits)

        scaled = self.table_values() / unit
        clipped = torch.clamp(scaled, self.entry_min, self.entry_max)

        return torch.round(clipped).detach() + (clipped - clipped.detach())

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def quantized_forward(self, x):
        """[N, P] layer inputs -> [N, O] quantized outputs."""

        # [N, P] LUT addresses, offset per input: row p * K + k of the table.
        index = quantize_indices(x, self.bits, *self.in_range) + self.offsets

        if self.pruning_stats is not None:
            self._accumulate_pruning_stats(index)

        # [O, P, K] -> [P * K, O]
        weight = self.table_entries().permute(1, 2, 0).reshape(
            self.n_in * self.levels,
            self.n_out
        )

        # Integer sum of P table entries per output (exact in float32).
        sums = F.embedding_bag(index, weight, mode="sum")

        # Rounding and saturation of the sum, as in the integer LUT model.
        n = sums.detach()

        if self.frac_bits > 0:
            n = torch.floor(
                (n + (1 << (self.frac_bits - 1))) / (1 << self.frac_bits)
            )

        n = torch.clamp(n, self.n_min, self.n_max)
        y = n * self.out_scale

        # Straight-through gradient: d(sum) / d(table entries) and the
        # input gradient of the SiLU branch; zero outside the output range.
        unit = self.out_scale / (1 << self.frac_bits)
        pre = sums * unit + self._input_gradient(x)

        clipped = torch.clamp(
            pre,
            self.n_min * self.out_scale,
            self.n_max * self.out_scale
        )

        return y + (clipped - clipped.detach())

    def _input_gradient(self, x):
        """
        Zero-valued term that carries d(output)/d(x) through the SiLU
        residual branch. Grad-Free Spline: no spline derivative.
        """

        if not (torch.is_grad_enabled() and x.requires_grad):
            return 0.0

        if self.pre_silu:
            x = F.silu(x)

        xq = quantize_values(x, self.bits, *self.in_range)

        weight = (self.u * self.connection_mask).detach()
        base = F.silu(xq) @ weight.t()

        return base - base.detach()

    # --------------------------------------------------------
    # Pruning statistics
    # --------------------------------------------------------

    def begin_pruning_stats(self):
        self.pruning_stats = {
            "counts": torch.zeros(
                self.n_in * self.levels,
                dtype=torch.float64,
                device=self.v.device
            ),
            "rows": 0,
        }

    def end_pruning_stats(self):
        self.pruning_stats = None

    @torch.no_grad()
    def _accumulate_pruning_stats(self, index):
        self.pruning_stats["counts"] += torch.bincount(
            index.reshape(-1),
            minlength=self.n_in * self.levels
        ).double()

        self.pruning_stats["rows"] += index.shape[0]

    @torch.no_grad()
    def connection_scores(self):
        """
        RMS contribution of every connection (o, p) on the calibration data.

        Contribution of one connection:

            c[n,o,p] = m[o,p] * ( spline_o(x[n,p]) / P + u[o,p] * SiLU(x[n,p]) )

        Every quantized input x[n,p] is one of the K table points, so

            mean_n c^2 = sum_k count[p,k] * T[o,p,k]^2 / N

        where count[p,k] is the number of calibration rows whose input p
        has address k, and T is the (unrounded) table.
        """

        stats = self.pruning_stats

        counts = stats["counts"].view(self.n_in, self.levels)
        table = self.table_values().double()

        mean_sq = (
            table * table * counts.unsqueeze(0)
        ).sum(dim=-1) / max(stats["rows"], 1)

        return torch.sqrt(mean_sq + 1e-12)

class KANELEKALayer(KANELEEdgeLayer):
    """KA layer (SaKAN + Grad-Free Spline) with QAT, a connection mask and integer tables."""

    def forward(self, x):
        # x: [N, P] -> [N, O]
        return self.quantized_forward(x)

class KANELEConv2dLayer(KANELEEdgeLayer):
    """
    KAonv layer (SaKANConv2dLayer + Grad-Free Spline) with QAT, a
    connection mask [O, P] over the patch elements and integer tables.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        grid_size=5,
        spline_order=3,
        bits=8,
        in_range=(-16, 16),
        out_range=(-16, 16),
        table_frac_bits=4
    ):
        super().__init__(
            in_channels * kernel_size * kernel_size,
            out_channels,
            grid_size,
            spline_order,
            bits,
            in_range,
            out_range,
            table_frac_bits
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        b = x.shape[0]

        # [B,C,H,W] -> [B,P,L]; zero padding is quantized like any input.
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        locations = patches.shape[-1]

        # [B,P,L] -> [B*L,P] -> [B*L,O]
        patches = patches.transpose(1, 2).reshape(b * locations, self.n_in)
        y = self.quantized_forward(patches)

        h = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1

        w = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1

        y = y.reshape(b, locations, self.out_channels).transpose(1, 2)

        return y.reshape(
            b,
            self.out_channels,
            h,
            w
        )