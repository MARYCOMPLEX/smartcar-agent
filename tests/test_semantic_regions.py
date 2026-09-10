import trimesh
import numpy as np
from smartcar.understanding.semantic_regions import separate_decorative_wheels


def test_only_paired_circular_side_components_are_removed():
    body=trimesh.creation.box([50,100,30]);body.apply_translation([0,0,25])
    wheels=[]
    for side in [-1,1]:
        m=trimesh.creation.cylinder(12,6,sections=64)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]));m.apply_translation([side*27,25,12]);wheels.append(m)
    result,report=separate_decorative_wheels(trimesh.util.concatenate([body]+wheels),.6)
    assert len(report["removed_components"])==2
    assert np.allclose(result.bounds,body.bounds)
    _,report2=separate_decorative_wheels(trimesh.util.concatenate([body,wheels[0]]),.6)
    assert len(report2["removed_components"])==0
