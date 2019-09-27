from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BetaTC:
    mi__γ_tc__λ_dw: bool = False
    γ_tc__λ_dw: bool = False
    kld__γmin1_tc: bool = False
    # kld__γmin1_tc__λmin1_dw

    def __post_init__(self):
        if sum((int(v) if isinstance(v, bool) else 2) for v in asdict(self).values()) != 1:
            raise ValueError
