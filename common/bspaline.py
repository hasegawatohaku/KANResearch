import torch

# ============================================================
# B-SPLINE
# ============================================================
# Builds a fixed, uniform knot vector for the B-spline basis.
# The vector is extended by `spline_order` intervals on both sides
# so that the basis is well-defined on the [xmin, xmax] range.
def make_knots(
    grid_size,
    spline_order,
    xmin=-3.0,
    xmax=3.0,
):
    """Create the fixed knot vector used by the B-spline basis"""

    step = (xmax - xmin) / grid_size

    knots = torch.arange(
        xmin - spline_order * step,
        xmax + spline_order * step + step * 0.5,
        step,
        dtype=torch.float32,
    )

    return knots

# Evaluates all B-spline basis functions at every element of x using the Cox-de Boor recursion.
# The right boundary is handled explicitly so that x == knots[-1] maps to the last basis function.
def bspline_basis(x, knots, order):
    """
    Evaluate all B-spline basis functions at x

    Input shape:
        arbitrary shape [...]

    Output shape:
        [..., n_spline]
    """

    x = x.unsqueeze(-1)

    # Degree-0 basis functions.
    basis = (
        (x >= knots[:-1]) &
        (x < knots[1:])
    ).float()

    # Cox-de Boor recursion.
    for k in range(1, order + 1):
        left_den = knots[k:-1] - knots[:-k-1]
        right_den = knots[k+1:] - knots[1:-k]

        left_num = x - knots[:-k-1]
        right_num = knots[k+1:] - x

        left = (left_num / left_den) * basis[..., :-1]
        right = (right_num / right_den) * basis[..., 1:]

        basis = left + right

    # Include x == right boundary.
    if basis.shape[-1] > 0:
        right_boundary = (x.squeeze(-1) == knots[-1]).unsqueeze(-1)

        boundary_basis = torch.cat(
            [
                torch.zeros_like(basis[..., :-1]),
                torch.ones_like(basis[..., -1:]),
            ],
            dim=-1,
        )

        basis = torch.where(
            right_boundary,
            boundary_basis,
            basis,
        )

    return basis