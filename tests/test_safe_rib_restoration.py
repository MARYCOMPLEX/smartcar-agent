from pathlib import Path
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,to_mesh,from_mesh
from smartcar.geometry.sdf import resample_designable_volume
from smartcar.geometry.repair import occupancy_mesh
from smartcar.structure.regularize import regularize_shell


def test_rib_restoration_cannot_reintroduce_an_exterior_thin_fin():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    outer=box([[-15,-20,0],[15,20,20]])+box([[14,4,0],[20,8,.6]])
    inner=box([[-12.6,-17.6,-1],[12.6,17.6,17.6]])
    body=outer-inner
    closure=dict(centers_xy=[],seam_z=-p.sliding_clearance,
                 connection_ribs=[dict(type='horizontal',center_xy=[13,6],length_mm=7,width_mm=2.4,z0_mm=0,z1_mm=6,angle_deg=0)])
    from smartcar.structure.closure_ribs import rib_solid
    cleaned,_=regularize_shell(body,closure,p)
    # The stronger final reconstruction now removes this fin even without a
    # safe envelope. The negative control is the original defect: restoring
    # the raw clipped rib AFTER manufacturing recreates the physical thin fin.
    assert not to_mesh(cleaned).contains([[18,6,.3]])[0]
    unsafe=cleaned+(rib_solid(closure['connection_ribs'][0])^body)
    assert to_mesh(unsafe).contains([[18,6,.3]])[0]
    volume,_=resample_designable_volume(to_mesh(outer),.6,p.nominal_wall)
    safe_envelope=from_mesh(occupancy_mesh(volume.distance>=p.minimum_wall,volume.origin,volume.pitch))
    safe,_=regularize_shell(body,closure,p,safe_envelope)
    assert not to_mesh(safe).contains([[18,6,.3]])[0]
    assert len(safe.decompose())==1 and to_mesh(safe).is_watertight
