import numpy as np
import trimesh
from smartcar.geometry.solid import box,cylinder,union,to_mesh
from smartcar.understanding.coordinate_frame import establish_frame


def test_rotated_single_component_car_uses_ground_contacts_for_up_axis():
    body=box([[-36,-90,16],[36,90,58]])
    wheels=[cylinder(18,x-6,x+6,(y,18),axis=0) for x in [-38,38] for y in [-57,49]]
    mesh=to_mesh(union([body]+wheels))
    assert len(mesh.split())==1
    transform=trimesh.transformations.euler_matrix(.7,-.4,1.1);transform[:3,3]=[170,-85,44]
    mesh.apply_transform(transform)
    normalized,record=establish_frame(mesh)
    np.testing.assert_allclose(normalized.extents,[88,180,58],atol=.5)
    assert record['selected']['score_components'].get('bilateral_ground_contact',0)>0
