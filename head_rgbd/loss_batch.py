"""Frame-dimension batch helpers for the per-frame loss terms.

P2 replaces the per-frame Python loss loop with these helpers.  The semantics
are those of ``loss_analysis/objectives.py`` and the legacy loops: each term is
an elementwise robust loss averaged over that frame's selected elements, then
weighted by the frame weight.  Nothing here changes a weight, denominator,
normalisation, robust beta, prior or ``frame_weight`` rule.
"""
from __future__ import annotations

import torch


def pack_correspond(correspond: dict, n: int, device) -> tuple | None:
    """Pad a {frame: (faces, bary, normal, target)} dict to fixed-shape tensors.

    Frames without a correspondence get an all-False mask row.  The returned
    ``K`` is the longest frame set, so the surface term stays a single batched
    operation.
    """
    if not correspond:
        return None
    length = max(int(len(row[0])) for row in correspond.values())
    if length <= 0:
        return None
    tri = torch.zeros((n, length, 3), dtype=torch.long, device=device)
    bary = torch.zeros((n, length, 3), device=device)
    normal = torch.zeros((n, length, 3), device=device)
    target = torch.zeros((n, length, 3), device=device)
    mask = torch.zeros((n, length), dtype=torch.bool, device=device)
    for index, (faces, bc, nrm, tgt) in correspond.items():
        count = int(len(faces))
        tri[index, :count] = faces
        bary[index, :count] = bc
        normal[index, :count] = nrm
        target[index, :count] = tgt
        mask[index, :count] = True
    return tri, bary, normal, target, mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row mean of ``values`` over the True elements of ``mask``.

    ``values`` is (n, ...) and ``mask`` (n, ...) or (n, k) for a trailing
    coordinate dimension.  Empty rows return 0, which reproduces the legacy
    ``if mask.any():`` skip (they carry weight 0 where they are used).
    """
    if values.dim() == mask.dim() + 1:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    dims = tuple(range(1, values.dim()))
    count = mask.sum(dim=dims)
    total = (values * mask).sum(dim=dims)
    return total / count.clamp(min=1)


def smooth_l1(values: torch.Tensor, beta: float) -> torch.Tensor:
    """Elementwise smooth-L1 with the zeros target used by the legacy loops."""
    return torch.nn.functional.smooth_l1_loss(
        values, torch.zeros_like(values), beta=beta, reduction="none")
