"""Common random numbers. Everything random in a BO run except the policy's own choices comes from
`stream(kind, step, replicate)`: a [n_tasks, ...] tensor from a generator seeded by those keys, indexed by
task row. Lanes of one task with the same replicate id see the same numbers whatever arm they run (CRN
across arms); different replicate ids see independent numbers (fresh nuisance for rollouts).
Replicate 0 is the evaluated trajectory.
"""

import torch

KINDS = {"shift": 1, "perturb": 2, "subset": 3, "fantasy": 4}


def _seed(base: int, kind: str, step: int, replicate: int) -> int:
    return (base * 1_000_003 + KINDS[kind] * 100_019 + step * 1_009 + replicate * 7_919_993) % (2**63 - 1)


def stream(
    base: int,
    kind: str,
    step: int,
    replicates: torch.Tensor,
    rows: torch.Tensor,
    shape: tuple[int, ...],
    n_tasks: int,
    normal: bool,
    device: str | torch.device,
) -> torch.Tensor:
    """[L, *shape] for lanes with task rows[l] and replicate ids replicates[l]."""
    out = torch.empty((len(rows), *shape))
    for rep in replicates.unique().tolist():
        g = torch.Generator().manual_seed(_seed(base, kind, step, int(rep)))
        full = torch.randn((n_tasks, *shape), generator=g) if normal else torch.rand((n_tasks, *shape), generator=g)
        sel = (replicates == rep).cpu()
        out[sel] = full[rows.cpu()[sel]]
    return out.to(device)
