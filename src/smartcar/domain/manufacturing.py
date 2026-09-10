from dataclasses import dataclass
from smartcar.io import read_json


@dataclass(frozen=True)
class ManufacturingProfile:
    values: dict

    @classmethod
    def load(cls, path):
        p = cls(read_json(path))
        assert p.nominal_wall >= p.minimum_wall >= 3 * p.nozzle_mm
        assert p.moving_clearance >= p.rigid_clearance > 0
        return p

    def __getattr__(self, name):
        try: return self.values[name]
        except KeyError: raise AttributeError(name)
