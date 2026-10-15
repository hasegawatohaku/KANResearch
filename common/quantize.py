import math
import torch

def quantization_params(bits, qmin, qmax):
    """Affine quantizer for [qmin, qmax]: scale and zero point (zero is a level)."""

    levels = 1 << bits
    scale = (qmax - qmin) / (levels - 1)

    zero_point = int(math.floor(-qmin / scale + 0.5))
    zero_point = min(max(zero_point, 0), levels - 1)

    return scale, zero_point


def quantize_indices(x, bits, qmin, qmax):
    """Values -> integer LUT addresses k in [0, 2**bits - 1]."""

    scale, zero_point = quantization_params(bits, qmin, qmax)

    index = torch.round(x.detach() / scale) + zero_point

    return index.clamp_(0, (1 << bits) - 1).long()


def dequantize_indices(index, bits, qmin, qmax):
    """LUT addresses k -> quantized values (k - zero_point) * scale."""

    scale, zero_point = quantization_params(bits, qmin, qmax)

    return (index.to(torch.float32) - zero_point) * scale


def quantization_grid(bits, qmin, qmax, device=None):
    """All quantized values in LUT address order: [K]."""

    index = torch.arange(1 << bits, device=device)

    return dequantize_indices(index, bits, qmin, qmax)


def quantize_values(x, bits, qmin, qmax):
    """
    Fake quantization with a clipped straight-through estimator (STE).

    Forward value: dequantize_indices(quantize_indices(x)), i.e. exactly
    the value of the LUT address of x.
    Backward: gradient 1 inside the representable range, 0 outside.
    """

    scale, zero_point = quantization_params(bits, qmin, qmax)

    low = -zero_point * scale
    high = ((1 << bits) - 1 - zero_point) * scale

    quantized = dequantize_indices(
        quantize_indices(x, bits, qmin, qmax),
        bits,
        qmin,
        qmax
    )

    clipped = torch.clamp(x, low, high)

    # (clipped - clipped.detach()) is exactly zero in the forward pass.
    return quantized.detach() + (clipped - clipped.detach())


def signed_bits(low, high):
    """Width of a two's complement integer that holds [low, high]."""

    low = int(low)
    high = int(high)

    negative = (-low - 1) if low < 0 else 0

    return max(max(high, 0).bit_length(), negative.bit_length()) + 1