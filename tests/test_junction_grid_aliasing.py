from pathlib import Path
import numpy as np
import pytest
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box, cylinder, to_mesh
from smartcar.structure.junction_cleanup import clean_closure_junctions
from smartcar.validation.geometry import wall_measurements


@pytest.mark.parametrize('angle,phase', [(31, .11), (53, .07)])
def test_diagonal_pillar_wall_has_no_sub_nozzle_grid_prongs(angle, phase, tmp_path, monkeypatch):
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    wall=box([[-p.nominal_wall/2,-12,0],[p.nominal_wall/2,12,12]]).rotate((0,11,0))
    pillar=cylinder(p.fastener_boss_radius,0,12,[-p.fastener_boss_radius,0])
    body=(wall+pillar).rotate((0,0,angle)).translate((phase,phase,0))
    b=to_mesh(body).bounds
    closure=dict(connection_ribs=[dict(type='vertical',center_xy=[0,0],radius_mm=16,z0_mm=float(b[0,2]),z1_mm=float(b[1,2]))])
    cleaned,record=clean_closure_junctions(body,closure,p)
    region=to_mesh(body).bounds+np.array([[-1]*3,[1]*3])
    measured=wall_measurements(to_mesh(cleaned),p,samples=0,regions=[region])
    assert measured['below_minimum']==0, (measured['minimum_mm'], measured['below_minimum'], measured['examples'])
    assert to_mesh(cleaned).is_watertight and len(cleaned.decompose())==1
    # Preserve the previous Gaussian-surface negative control explicitly; the
    # new distance-to-balls surface no longer creates that same fragile ledge.
    from smartcar.geometry.repair import occupancy_mesh
    with monkeypatch.context() as patch:
        patch.setattr('smartcar.geometry.repair.rolling_sphere_mesh',
            lambda mask,origin,pitch,diameter:occupancy_mesh(mask,origin,pitch,smoothing_radius_mm=p.nozzle_mm))
        legacy,_=clean_closure_junctions(body,closure,p)
    legacy=legacy.simplify(p.numerical_tolerance/2)
    assert wall_measurements(to_mesh(legacy),p,samples=0,regions=[region])['below_minimum']>0
    # Include the formal finalization and physical STL roundtrip, not just an
    # intermediate body which a later generic simplifier could invalidate.
    import trimesh
    from smartcar.geometry.print_mesh import finalize_print_parts
    final,records=finalize_print_parts({'body':cleaned},p)
    assert records['body'].get('status','PASS')!='FAIL'
    path=tmp_path/'body.stl';to_mesh(final['body']).export(path)
    exported=trimesh.load(path,force='mesh')
    assert exported.is_watertight and exported.is_winding_consistent
    assert exported.nondegenerate_faces().all()
    from smartcar.geometry.mesh import mesh_stats
    assert mesh_stats(exported,include_components=False)['degenerate_triangles']==0
    assert wall_measurements(exported,p,samples=0,regions=[region])['below_minimum']==0
