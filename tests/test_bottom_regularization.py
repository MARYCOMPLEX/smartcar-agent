from pathlib import Path
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,cylinder,to_mesh,intersection_volume
from smartcar.structure.bottom_regularization import regularize_bottom


def test_thin_edge_web_removed_without_refilling_access_or_moving_mount():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    base=box([[-20,-25,0],[20,25,2.4]])+box([[19,4,0],[28,5,2.4]])
    opening=cylinder(3,-1,4,(0,0))
    mount=cylinder(3.2,2.3,18,(12,12))
    result,record=regularize_bottom((base-opening)+mount,2.4,p)
    assert record['removed_base_mm3']>0
    assert not to_mesh(result).contains([[25,4.5,1.2]])[0]
    assert intersection_volume(result,opening)<p.collision_volume_tolerance
    assert intersection_volume(result,mount)>=mount.volume()-p.collision_volume_tolerance
    assert len(result.decompose())==1
