from pathlib import Path
import numpy as np
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,cylinder,to_mesh,intersection_volume
from smartcar.structure.junction_cleanup import clean_closure_junctions
from smartcar.validation.geometry import wall_measurements


def test_small_junction_prong_is_detected_and_removed_without_random_sampling():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    # A vertical, sub-nozzle planar prong on a much larger valid wall, rotated
    # off the voxel axes. No source-car coordinates or historical pose is used.
    main=box([[-8,-8,0],[8,0,12]])
    fin=box([[3,-.1,0],[3.3,1.1,12]])
    body=(main+fin).rotate((0,0,31))
    closure=dict(connection_ribs=[dict(type='vertical',center_xy=[0,0],radius_mm=8,z0_mm=0,z1_mm=12)])
    region=to_mesh(body).bounds+np.array([[-1]*3,[1]*3])
    before=wall_measurements(to_mesh(body),p,samples=0,regions=[region])
    assert before['critical_feature_samples']>0 and before['below_minimum']>0
    cleaned,record=clean_closure_junctions(body,closure,p)
    after=wall_measurements(to_mesh(cleaned),p,samples=0,regions=[region])
    assert record['removed_mm3']>0 and after['below_minimum']==0
    assert len(cleaned.decompose())==1 and to_mesh(cleaned).is_watertight
    # Reconstructing one stable grid surface has bounded boundary uncertainty;
    # it must not grow arbitrary bulk while fixing a small feature.
    import trimesh
    distance=trimesh.proximity.signed_distance(to_mesh(body),to_mesh(cleaned).vertices)
    assert max(0.,-distance.min())<=np.sqrt(3)*p.shell_regularization_pitch/2+p.numerical_tolerance
    safe_probe=box([[-2,-6,2],[2,-2,10]]).rotate((0,0,31))
    assert intersection_volume(cleaned,safe_probe)>=safe_probe.volume()-p.collision_volume_tolerance
