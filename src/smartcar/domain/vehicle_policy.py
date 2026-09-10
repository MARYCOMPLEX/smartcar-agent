"""Configurable vehicle design policy, separate from physical tolerances."""
from dataclasses import dataclass, asdict
from smartcar.io import read_json


@dataclass(frozen=True)
class VehiclePolicy:
    minimum_wheelbase_fraction: float = 0.42
    preferred_wheelbase_fraction: float = 0.62
    maximum_wheelbase_fraction: float = 0.82
    minimum_tire_gap_diameters: float = 0.35
    source_axle_tolerance_fraction: float = 0.05
    minimum_source_wheelbase_retention: float = 0.85
    maximum_overhang_imbalance_fraction: float = 0.22
    contact_section_height_fractions: tuple = (0.04, 0.08, 0.12, 0.16, 0.20)
    contact_lateral_fraction: float = 0.24
    contact_bins: int = 400
    contact_gap_fraction: float = 0.012
    minimum_contact_band_fraction: float = 0.02
    maximum_contact_band_fraction: float = 0.36
    contact_center_agreement_fraction: float = 0.035
    minimum_agreeing_sections: int = 2
    maximum_wheelwell_addition_fraction: float = 0.12
    wheelwell_radius_margin_fraction: float = 0.12
    belly_sample_width_fraction: float = 0.60
    belly_height_quantile: float = 0.60
    preferred_source_tire_scale: float = 1.0
    minimum_source_tire_scale: float = 0.60
    source_tire_scale_is_hard: bool = False

    @classmethod
    def load(cls, path):
        return cls(**read_json(path))

    def to_dict(self):
        return asdict(self)
