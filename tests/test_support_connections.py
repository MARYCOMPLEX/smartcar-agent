from pathlib import Path
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,cylinder,to_mesh,union,intersection_volume
from smartcar.structure.support_connections import connect_tray_supports


def test_floating_mount_gets_connected_without_covering_access_slot():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    floor=p.bottom_thickness
    base=box([[-20,-25,0],[20,25,floor]])
    support=cylinder(3.2,floor-.1,18,(28,0))
    slot=box([[22,-1,-1],[24,1,5]])
    vehicle=to_mesh(box([[-32,-30,0],[32,30,40]]))
    result,records=connect_tray_supports(base+support,vehicle,floor,slot,p)
    assert records and len(result.decompose())==1
    assert intersection_volume(result,slot)<p.collision_volume_tolerance
    assert intersection_volume(result,base)>=base.volume()-p.collision_volume_tolerance
    assert intersection_volume(result,support)>=support.volume()-p.collision_volume_tolerance


def test_connection_cannot_cross_a_complete_wheel_or_access_barrier():
    import pytest
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    base=box([[-20,-25,0],[20,25,2.4]]);support=cylinder(3.2,2.3,18,(28,0))
    barrier=box([[21,-40,-2],[24,40,25]])
    with pytest.raises(ValueError,match='SUPPORT_CONNECTION_NO_COLLISION_FREE_RAIL'):
        connect_tray_supports(base+support,to_mesh(box([[-35,-35,0],[35,35,40]])),2.4,barrier,p)


def test_support_rail_preserves_measured_lid_sliding_clearance():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    floor=p.bottom_thickness
    base=box([[-20,-25,0],[20,0,floor]])+box([[-20,0,0],[19,25,floor]])
    support=cylinder(3.2,floor-.1,18,(28,0))
    vehicle=to_mesh(box([[-35,-35,0],[35,35,40]]))
    result,records=connect_tray_supports(base+support,vehicle,floor,union([]),p)
    # Only the free span lies under this independent shell segment. The base
    # and original support separately clear it, isolating the generated bridge.
    lid=box([[21,-1,floor+p.sliding_clearance],[24,1,8]])
    assert min(base.min_gap(lid,1),support.min_gap(lid,1))>=p.sliding_clearance
    assert len(result.decompose())==1 and records
    assert result.min_gap(lid,1)+p.numerical_tolerance>=p.sliding_clearance


def test_support_starting_exactly_at_floor_joins_through_a_measured_face():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    floor=p.bottom_thickness
    base=box([[-20,-25,0],[20,0,floor]])+box([[-20,0,0],[19,25,floor]])
    support=cylinder(3.2,floor,18,(28,0))
    vehicle=to_mesh(box([[-35,-35,0],[35,35,40]]))
    result,records=connect_tray_supports(base+support,vehicle,floor,union([]),p)
    assert len(result.decompose())==1 and to_mesh(result).is_watertight
    assert intersection_volume(result,support)>=support.volume()-p.collision_volume_tolerance
    assert records[0]['island_attachment']['volume_mm3']<=p.collision_volume_tolerance
    assert records[0]['island_attachment']['shared_face_area_mm2']>=p.minimum_wall**2
    assert records[0]['bounds_mm'][1][2]<=floor+p.numerical_tolerance


def test_edge_contact_does_not_count_as_structural_attachment():
    from smartcar.structure.support_connections import attachment_measurement
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    a=box([[0,0,0],[5,5,2.4]])
    b=box([[5,0,2.4],[10,5,8]])
    assert attachment_measurement(a,b,p) is None
