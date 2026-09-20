"""Additive P-v2 selector: legal proposal choice is independent of score sign."""
from __future__ import annotations

from .cw1_sr0_structure_v1 import enumerate_superregion_bank, superregion_sha256
from .dino_rcde_sr0_mt_p_runtime_v1 import DeployableDirectionOutput
from .dino_rcde_sr0_mt_semantic_h0_v1 import (
    StructuralProposal,
    select_structural_proposal,
)


def structural_proposal_from_deployable_output(
    output: DeployableDirectionOutput,
) -> StructuralProposal:
    if not isinstance(output, DeployableDirectionOutput):
        raise TypeError("P-v2 selector requires DeployableDirectionOutput")
    rows = tuple(
        enumerate_superregion_bank(
            output.record.colnomic_query_grid_shape,
            include_r0_control=False,
        )
    )
    hashes = tuple(superregion_sha256(row) for row in rows)
    if len(hashes) != output.row_scores.numel():
        raise RuntimeError("P-v2 row-bank population drift")
    return select_structural_proposal(
        output.row_scores,
        output.row_ready,
        hashes,
    )


__all__ = ["structural_proposal_from_deployable_output"]

