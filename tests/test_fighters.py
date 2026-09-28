import numpy as np
import pandas as pd

from sparring.config import FighterConfig
from sparring.perception import appearance
from sparring.perception.fighters import seed_pair, select_fighters

H = 1920


def _desc(bin_idx: int, noise: float, rng) -> np.ndarray:
    d = np.zeros(appearance.DESCRIPTOR_SIZE, np.float32)
    n = appearance.BINS_PER_REGION
    for r in range(len(appearance.REGIONS)):
        d[r * n + bin_idx] = 1.0
        d[r * n : (r + 1) * n] += noise * rng.random(n)
        d[r * n : (r + 1) * n] /= d[r * n : (r + 1) * n].sum()
    return d


def _scene(swap_ids: bool, with_bystander: bool = True):
    """Two fighters cross paths; optionally the tracker swaps their IDs at the crossing.
    A small bystander wearing the same colours as fighter P stands in the background."""
    rng = np.random.default_rng(0)
    rows, desc = [], []
    for f in range(60):
        px = 300 + f * 400 / 59
        qx = 700 - f * 400 / 59
        pid, qid = (2, 1) if swap_ids and f >= 30 else (1, 2)
        for x, tid, colour in ((px, pid, 0), (qx, qid, 40)):
            rows.append((f, tid, x - 150, 600, x + 150, 1500, 0.9))
            desc.append(_desc(colour, 0.01, rng))
        if with_bystander:
            rows.append((f, 9, 950, 500, 1010, 700, 0.8))
            desc.append(_desc(0, 0.01, rng))
    dets = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
    return dets, np.array(desc)


def test_seed_pair_prefers_large_close_people():
    dets, _ = _scene(swap_ids=False)
    assert seed_pair(dets, H) == (1, 2)


def test_identity_survives_tracker_id_swap():
    dets, desc = _scene(swap_ids=True)
    sel = select_fighters(dets, desc, H, FighterConfig())
    a = sel.assignments[sel.assignments.fighter == "A"].sort_values("frame")
    b = sel.assignments[sel.assignments.fighter == "B"].sort_values("frame")
    assert len(a) == 60 and len(b) == 60
    # A started on the left and walks right the whole time, whatever the tracker ID says.
    centres = ((a.x1 + a.x2) / 2).to_numpy()
    assert np.all(np.diff(centres) > 0)
    assert 9 not in set(a.track_id) | set(b.track_id)  # bystander never picked


def test_override_chooses_given_tracks():
    dets, desc = _scene(swap_ids=False)
    sel = select_fighters(dets, desc, H, FighterConfig(), override=(2, 1))
    assert sel.seed_tracks == {"A": 1, "B": 2}  # A = whichever is further left


def test_appearance_distance_bounds():
    rng = np.random.default_rng(1)
    a, b = _desc(0, 0.0, rng), _desc(40, 0.0, rng)
    assert appearance.distance(a, a) < 1e-6
    assert abs(appearance.distance(a, b) - 1.0) < 1e-6
