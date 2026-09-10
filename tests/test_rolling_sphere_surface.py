from pathlib import Path
import numpy as np
import pytest
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,cylinder,to_mesh
from smartcar.geometry.voxelize import voxelize_closed_solid
from smartcar.geometry.repair import rolling_sphere_mesh
from smartcar.validation.geometry import wall_measurements


@pytest.mark.parametrize('gap',[.15,.45,.75])
def test_distance_to_balls_does_not_blur_separate_walls_into_a_thin_bridge(gap):
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    source=cylinder(3,0,10,[-3-gap/2,0])+cylinder(3,0,10,[3+gap/2,0])+box([[-8,-4,0],[8,4,2.4]])
    cells,origin,_=voxelize_closed_solid(to_mesh(source),p.shell_regularization_pitch)
    mesh=rolling_sphere_mesh(cells,origin,p.shell_regularization_pitch,p.nominal_wall)
    measurements=wall_measurements(mesh,p,samples=0,regions=[mesh.bounds+np.array([[-1]*3,[1]*3])])
    assert measurements['below_minimum']==0
    assert mesh.is_watertight and len(mesh.split())==1
    assert not mesh.contains([[0,0,7]])[0]
