from pathlib import Path
import numpy as np
import trimesh
from smartcar.geometry.sdf import DesignableVolume,resample_designable_volume
from smartcar.domain.manufacturing import ManufacturingProfile


def test_scaling_does_not_erase_a_narrow_valid_wheel_ground_interval():
    profile=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    exterior=trimesh.creation.box([15,30,15]);exterior.apply_translation([0,0,7.5])
    base,_=resample_designable_volume(exterior,.6,profile.nominal_wall)
    coarse=DesignableVolume(base.occupancy,base.origin*4,base.pitch*4,base.wall)
    physical,_=resample_designable_volume(exterior.copy().apply_scale(4),.6,profile.nominal_wall)
    extents=[18,54,22]
    coarse_centers,_=coarse.feasible_centers(extents,profile.rigid_clearance,bottom=3.,stride=1)
    precise_centers,_=physical.feasible_centers(extents,profile.rigid_clearance,bottom=3.,stride=1)
    # The CAD shaft is centered vertically in this 22 mm-high housing, and
    # the fixed wheel radius is 18.5 mm. No hand-edited placement is supplied.
    max_shaft_z=18.5-profile.rigid_clearance
    assert not np.any(coarse_centers[:,2]<=max_shaft_z)
    assert np.any(precise_centers[:,2]<=max_shaft_z)
    assert physical.pitch==profile.voxel_pitch
    assert physical.error<coarse.error
