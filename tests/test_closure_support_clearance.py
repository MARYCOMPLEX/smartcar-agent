from pathlib import Path
import numpy as np
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,cylinder,to_mesh,intersection_volume
from smartcar.structure.closure import screw_closure


def test_restored_screw_boss_cannot_consume_axle_bracket_clearance():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    floor=p.bottom_thickness
    exterior=box([[-20,-24,0],[20,24,24]])
    base=box([[-20,-24,0],[20,24,floor]])
    bracket_bounds=np.array([[-11.85,-20,floor],[-9.85,-18,14.]])
    bracket=box(bracket_bounds)
    pocket=box(bracket_bounds+np.array([[-1]*3,[1]*3])*p.sliding_clearance)
    body=box([[-20,-24,floor+p.sliding_clearance],[20,24,24]])-pocket
    # The old corner preference accepts this partly embedded cylinder, then
    # restores it into the pocket. The hardware can be clear while this support
    # is only 0.15 mm away, less than the unchanged 0.35 mm sliding requirement.
    bad=cylinder(p.fastener_boss_radius,floor+p.sliding_clearance,
                 floor+p.sliding_clearance+p.screw_fit['thread_length'],(-16,-20))
    assert intersection_volume(bad,body)>bad.volume()*.98
    assert bad.min_gap(bracket,1)<p.sliding_clearance-p.numerical_tolerance
    _,lid,record=screw_closure(body,base+bracket,[],to_mesh(exterior),floor,p)
    assert record['screw_count']==4
    for xy in record['centers_xy']:
        native=cylinder(p.fastener_boss_radius,floor+p.sliding_clearance,
                        floor+p.sliding_clearance+p.screw_fit['thread_length'],xy)
        assert native.min_gap(lid,1)+p.numerical_tolerance>=p.sliding_clearance
