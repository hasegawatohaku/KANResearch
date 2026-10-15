import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.pruning import kanele_layers
from common.quantize import quantize_indices, signed_bits, quantization_params
from common.layers.kanele import KANELEKALayer, KANELEConv2dLayer

class LUTEdgeLayer(nn.Module):
    """
    Integer inference layer.

    weight[p * K + k, o] = E[o, p, k] (int32). For one input row,

        S[o] = sum_p weight[p * K + k_p, o]                (int64)
        n[o] = clamp( floor( (S[o] + 2**(F-1)) / 2**F ), -zp, K-1-zp )

    and the output value is n[o] * out_scale. No B-spline basis, no SiLU
    and no floating-point multiply-accumulate is evaluated.

    chunk_elements: maximum number of table values [rows, P, O] gathered
    at once in lookup(); limits memory, does not change the result.
    """

    @torch.no_grad()
    def __init__(self, layer, chunk_elements):
        super().__init__()

        self.chunk_elements = int(chunk_elements)

        self.n_in = layer.n_in
        self.n_out = layer.n_out
        self.bits = layer.bits
        self.levels = layer.levels
        self.in_range = layer.in_range
        self.out_range = layer.out_range
        self.frac_bits = layer.frac_bits
        self.pre_silu = layer.pre_silu
        self.out_scale = layer.out_scale
        self.out_zero_point = layer.out_zero_point
        self.n_min = layer.n_min
        self.n_max = layer.n_max

        # [O, P, K] integer entries -> [P * K, O]
        entries = torch.round(layer.table_entries()).to(torch.int32)

        self.register_buffer(
            "weight",
            entries.permute(1, 2, 0)
                   .reshape(self.n_in * self.levels, self.n_out)
                   .contiguous()
        )

        self.register_buffer("offsets", layer.offsets.detach().clone())

        self.register_buffer(
            "connection_mask",
            layer.connection_mask.detach().clone()
        )

    def table(self):
        """Integer LUT contents as [O, P, K]."""

        return self.weight.reshape(
            self.n_in,
            self.levels,
            self.n_out
        ).permute(2, 0, 1)

    def hardware_widths(self):
        """Signed widths (bits) of the table entries and of the accumulator."""

        table = self.table().to(torch.int64)

        entry_bits = signed_bits(table.min().item(), table.max().item())

        # Largest possible |S[o]| over all addresses, plus the rounding constant.
        worst_sum = table.abs().amax(dim=-1).sum(dim=-1).max().item()

        if self.frac_bits > 0:
            worst_sum += 1 << (self.frac_bits - 1)

        accumulator_bits = signed_bits(-worst_sum, worst_sum)

        return entry_bits, accumulator_bits

    @torch.no_grad()
    def lookup(self, x):
        """[N, P] inputs -> [N, O] quantized outputs."""

        out = torch.empty(
            x.shape[0],
            self.n_out,
            dtype=torch.float32,
            device=x.device
        )

        rows = max(1, self.chunk_elements // (self.n_in * self.n_out))

        for start in range(0, x.shape[0], rows):
            index = quantize_indices(
                x[start:start + rows],
                self.bits,
                *self.in_range
            ) + self.offsets

            # [rows, P, O] table lookups -> integer sums [rows, O]
            acc = self.weight[index].sum(dim=1, dtype=torch.int64)

            if self.frac_bits > 0:
                acc = torch.div(
                    acc + (1 << (self.frac_bits - 1)),
                    1 << self.frac_bits,
                    rounding_mode="floor"
                )

            acc = torch.clamp(acc, self.n_min, self.n_max)

            out[start:start + rows] = acc.to(torch.float32) * self.out_scale

        return out

class LUTKALayer(LUTEdgeLayer):
    """LUT version of KANELEKALayer: [N, P] -> [N, O]."""

    @torch.no_grad()
    def forward(self, x):
        return self.lookup(x)

class LUTKAConv2dLayer(LUTEdgeLayer):
    """LUT version of KANELEConv2dLayer: unfold -> table lookups -> integer sum."""

    def __init__(self, layer, chunk_elements):
        super().__init__(layer, chunk_elements)

        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding

    @torch.no_grad()
    def forward(self, x):
        b = x.shape[0]

        # [B,C,H,W] -> [B,P,L]
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        locations = patches.shape[-1]

        # [B,P,L] -> [B*L,P] -> [B*L,O]
        patches = patches.transpose(1, 2).reshape(b * locations, self.n_in)
        y = self.lookup(patches)

        h = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1

        w = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1

        y = y.reshape(b, locations, self.out_channels).transpose(1, 2)

        return y.reshape(
            b,
            self.out_channels,
            h,
            w
        )

 
@torch.no_grad()
def build_kanele_lut_model(model, chunk_elements):
    """
    Copy the trained ALL U-KAN + KANELE model and replace every KA / KAonv
    layer with its integer LUT layer. Pooling, upsampling and skip
    connections are kept.
    """

    model.eval()

    lut_model = copy.deepcopy(model)

    for parent in list(lut_model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, KANELEConv2dLayer):
                setattr(parent, child_name, LUTKAConv2dLayer(child, chunk_elements))
            elif isinstance(child, KANELEKALayer):
                setattr(parent, child_name, LUTKALayer(child, chunk_elements))

    # Only integer lookup tables remain: no trainable parameter.
    assert len(kanele_layers(lut_model)) == 0
    assert sum(p.numel() for p in lut_model.parameters()) == 0

    return lut_model.eval()


def kanele_lut_widths(lut_model):
    """Maximum entry width and accumulator width over all LUT layers."""

    widths = [
        module.hardware_widths()
        for module in lut_model.modules()
        if isinstance(module, LUTEdgeLayer)
    ]

    return (
        max(entry for entry, _ in widths),
        max(accumulator for _, accumulator in widths)
    )


def export_kanele_lut(
        lut_model, 
        path,
        encoder_channels,
        pools,
        ka_widths):
    """
    Save all integer lookup tables and quantizer metadata as JSON
    (KANELÉ [6], Sec. 4.1.2: per-connection truth tables in JSON files).

    As in truth_tables.json of the KANELÉ code, every connection is
    stored under the key "{input}_{output}" with its activity flag and
    its integer table values; the input index of a KAonv layer is the
    patch element p of torch.nn.functional.unfold.
    """

    layers = []

    for name, module in lut_model.named_modules():
        if not isinstance(module, LUTEdgeLayer):
            continue

        in_scale, in_zero_point = quantization_params(
            module.bits,
            *module.in_range
        )
        entry_bits, accumulator_bits = module.hardware_widths()

        # [O, P, K] integers and [O, P] activity flags as Python lists.
        table = module.table().detach().cpu().tolist()
        mask = module.connection_mask.detach().cpu().tolist()

        truth_tables = {
            f"{p}_{o}": {
                "active": int(mask[o][p] > 0.5),
                "values_int": table[o][p],
            }
            for o in range(module.n_out)
            for p in range(module.n_in)
        }

        entry = {
            "name": name,
            "type": "KAonv" if isinstance(module, LUTKAConv2dLayer) else "KA",
            "input_features": module.n_in,
            "output_features": module.n_out,
            "bits": module.bits,
            "input_scale": in_scale,
            "input_zero_point": in_zero_point,
            "output_scale": module.out_scale,
            "output_zero_point": module.out_zero_point,
            "table_frac_bits": module.frac_bits,
            "entry_bits": entry_bits,
            "accumulator_bits": accumulator_bits,
            "pre_silu": module.pre_silu,
            "truth_tables": truth_tables
        }

        if isinstance(module, LUTKAConv2dLayer):
            entry.update({
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding
            })

        layers.append(entry)

    payload = {
        "format": "KANResearch-ALLUKAN-KANELE-LUT-v2",
        "description": (
            "Integer truth tables per connection: truth_tables['{p}_{o}']"
            "['values_int'][k] = E[o, p, k]. Input address: "
            "k = clamp(round(x / input_scale) + input_zero_point, 0, 2**bits - 1). "
            "Output: S = sum over active p of E[o, p, k_p] (patch elements of "
            "torch.nn.functional.unfold for KAonv); "
            "n = clamp(floor((S + 2**(F-1)) / 2**F), -output_zero_point, "
            "2**bits - 1 - output_zero_point) with F = table_frac_bits "
            "(n = S for F = 0); output value = n * output_scale."
        ),
        "encoder_channels": list(encoder_channels),
        "pools": list(pools),
        "ka_widths": list(ka_widths),
        "layers": layers
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
