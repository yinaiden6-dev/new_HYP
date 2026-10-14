"""A single frozen input-scale change to the existing pre-LLM adapter.

No output-head M channel, extra parameters, changed token normalization, or
changed residual scale. The constant-condition function is exactly unchanged.
"""
import math
from rc_prellm_m_adapter_v1 import QualityResidualAdapter


class ScaledQualityResidualAdapter(QualityResidualAdapter):
    def __init__(self, *args, condition_gain, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(condition_gain) or condition_gain <= 0:
            raise ValueError('positive finite condition_gain required')
        # The authority pins this scalar; it is not a learned model parameter.
        self.condition_gain = float(condition_gain)

    def standardized_mass(self, mass, tokens):
        return super().standardized_mass(mass, tokens) * self.condition_gain
