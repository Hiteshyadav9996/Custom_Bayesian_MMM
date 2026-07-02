from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToyConfig:
    seed: int = 20260702
    n_weeks: int = 260
    decay_length: int = 16
    train_weeks: int = 208
    training_epochs: int = 1024
    snapshot_interval: int = 50
    regions: tuple[str, ...] = ("north", "south", "east", "west", "central")
    brands: tuple[str, ...] = ("brand_a", "brand_b", "brand_c", "brand_d", "brand_e")
    channels: tuple[str, ...] = ("off_trade", "on_trade")
    generic_features: tuple[str, ...] = ("distribution", "temperature", "macro")
    vehicles: tuple[str, ...] = (
        "search_continuous",
        "social_continuous",
        "tv_sparse",
        "promo_sparse",
        "sponsorship_one_time",
    )
    dark_blocks: tuple[tuple[int, int], ...] = ((16, 40), (80, 104), (144, 168), (208, 232))
    pricing_exponent_bounds: tuple[float, float] = (0.5, 5.0)
    baseline_prior: float = 105.0
    feature_beta_prior: tuple[float, ...] = (0.055, 0.035, 0.030)
    pricing_offset_prior: float = 0.65
    pricing_exponent_prior: float = 2.1
    alpha_prior: float = 0.78
    roi_vehicle_prior: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "asymptote": (0.80, 1.10, 1.45, 0.90, 1.85),
            "slope": (1.45, 1.20, 0.70, 1.65, 0.50),
            "beta": (0.18, 0.26, 0.50, 0.10, 0.64),
            "gamma": (0.42, 0.55, 0.76, 0.28, 0.90),
        }
    )

    @property
    def n_series(self) -> int:
        return len(self.regions) * len(self.brands) * len(self.channels)

    @property
    def no_investment_weeks(self) -> tuple[int, ...]:
        return tuple(week for start, end in self.dark_blocks for week in range(start, end))


def default_config() -> ToyConfig:
    return ToyConfig()
