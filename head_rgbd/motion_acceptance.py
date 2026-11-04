"""Share expensive mesh evaluations while retaining per-frame fixed-ray gates."""


def batch_backtrack(frame_ids, evaluate_scale, accept_frame):
    """Select each frame's first accepted scale; unresolved frames retain scale 0.

    Frames must be independent in evaluate_scale (as in FLAME's batch axis).
    accept_frame returns the original acceptance decision and diagnostic fields.
    """
    pending = list(frame_ids)
    selected = {}
    evaluations = 0
    for scale in (1., .5, .25, .125, 0.):
        if not pending:
            break
        candidates = evaluate_scale(scale)
        evaluations += 1
        remaining = []
        for frame in pending:
            info = accept_frame(frame, candidates[frame])
            if info['accepted'] or scale == 0.:
                selected[frame] = dict(info, scale=scale)
            else:
                remaining.append(frame)
        pending = remaining
    return selected, evaluations
