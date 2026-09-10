import json
from pathlib import Path
import numpy as np
import pytest
from smartcar.domain.hardware import intrinsic_rigid_bounds


def test_nominal_dimensions_locate_rigid_body_without_flexible_tail():
    raw={"id":"synthetic_pack","role":"battery","rigid_body":{"envelope_xyz_mm":[50,30,16]}}
    features={"planes":[{"area":400,"bounds":[[-15,0,-8],[15,0,8]]},{"area":390,"bounds":[[-15,50,-8],[15,50,8]]}]}
    b=intrinsic_rigid_bounds(raw,features,[[-15,0,-8],[15,160,10]])
    assert np.allclose(b,[[-15,0,-8],[15,50,8]])


def test_no_nominal_match_fails_instead_of_clipping_arbitrarily():
    raw={"id":"unknown","role":"drive_unit","rigid_body":{"envelope_xyz_mm":[50,30,16]}}
    with pytest.raises(ValueError,match="RIGID_ENVELOPE_NOT_LOCATED"):
        intrinsic_rigid_bounds(raw,{"planes":[]},[[0,0,0],[100,40,30]])
