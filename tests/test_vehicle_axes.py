import numpy as np
import trimesh
from smartcar.geometry.solid import cylinder,to_mesh
from smartcar.understanding.coordinate_frame import establish_frame
from smartcar.understanding.semantic_regions import separate_decorative_wheels


def tall_vehicle():
    body=trimesh.creation.box([30,80,48]);body.apply_translation([0,0,29])
    axles=[]
    for y in [-25,25]:
        solid=cylinder(3,-17,17,(y,12),axis=0)
        for side in [-1,1]:
            solid+=cylinder(12,side*16-5,side*16+5,(y,12),axis=0)
        axles.append(to_mesh(solid))
    return body,axles


def test_tall_vehicle_uses_bilateral_axles_not_shortest_dimension_as_height():
    body,axles=tall_vehicle();source=trimesh.util.concatenate([body]+axles)
    transform=trimesh.transformations.rotation_matrix(.81,[2,1,3]);transform[:3,3]=[31,-17,9]
    source.apply_transform(transform)
    normalized,frame=establish_frame(source)
    assert np.allclose(normalized.extents,[42,80,53],atol=.1)
    recovered_up=frame['input_to_vehicle'][:3,:3]@transform[:3,2]
    assert np.allclose(recovered_up,[0,0,1],atol=1e-5)
    assert frame['selected']['grounded_wheel_count']==4


def test_connected_circular_axle_pairs_are_removed_without_removing_body():
    body,axles=tall_vehicle()
    remaining,record=separate_decorative_wheels(trimesh.util.concatenate([body]+axles),.6)
    assert len(record['removed_components'])==2
    assert all(c['type']=='connected_axle_pair' for c in record['removed_components'])
    assert np.allclose(remaining.bounds,body.bounds)
    assert np.isclose(remaining.volume,body.volume)


def test_symmetric_rectangular_component_is_not_misidentified_as_axle():
    body,_=tall_vehicle()
    rectangular=trimesh.creation.box([42,24,24]);rectangular.apply_translation([0,25,12])
    source=trimesh.util.concatenate([body,rectangular])
    remaining,record=separate_decorative_wheels(source,.6)
    assert not record['removed_components']
    assert np.isclose(remaining.volume,source.volume)
