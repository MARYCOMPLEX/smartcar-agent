import numpy as np
import trimesh
from smartcar.domain.vehicle_policy import VehiclePolicy
from smartcar.understanding.axle_anchors import detect_axle_anchors, horizontal_segments


def contact_car():
    body=trimesh.creation.box([72,180,42]);body.apply_translation([0,0,37])
    pieces=[body]
    for x in [-38,38]:
        for y in [-57,49]:
            wheel=trimesh.creation.cylinder(radius=18,height=12,sections=48)
            wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
            wheel.apply_translation([x,y,18]);pieces.append(wheel)
    return trimesh.util.concatenate(pieces)


def test_sections_recover_axles_without_component_labels():
    # A raw combined triangle soup has no wheel labels. The section method must
    # measure both axles, even if a component classifier misses every wheel.
    mesh=contact_car();e=detect_axle_anchors(mesh)
    assert e['confidence']=='measured'
    np.testing.assert_allclose([a['y_mm'] for a in e['axles']],[-57,49],atol=1.0)
    assert 103<e['source_wheelbase_mm']<109


def test_flat_box_does_not_invent_source_wheels():
    e=detect_axle_anchors(trimesh.creation.box([80,190,70]))
    assert e['confidence']=='fallback' and e['axles']==[]


def test_section_evidence_scales_and_translates():
    mesh=contact_car();original=detect_axle_anchors(mesh)
    mesh.apply_scale(1.7);mesh.apply_translation([40,-130,25])
    moved=detect_axle_anchors(mesh)
    np.testing.assert_allclose([a['y_mm'] for a in moved['axles']],np.array([a['y_mm'] for a in original['axles']])*1.7-130,atol=.01)


def test_horizontal_section_edges_have_exact_height():
    segments=horizontal_segments(contact_car(),8.7)
    assert len(segments)>0
    np.testing.assert_allclose(segments[:,:,2],8.7,atol=1e-10)
