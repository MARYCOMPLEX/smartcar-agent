from pathlib import Path
import numpy as np
import pytest
import trimesh
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.errors import GeometryInputError
from smartcar.geometry.repair import occupancy_mesh,repair_volume,check_voxel_budget
from smartcar.understanding.input_scale import prepare_appearance


def profile():return ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')


def test_tiny_unlabelled_source_reports_units_before_marching_cubes():
    with pytest.raises(GeometryInputError,match='INPUT_SCALE_TOO_SMALL'):
        prepare_appearance(trimesh.creation.box([.55,1.2,.28]),profile())


def test_target_length_and_source_transform_work_after_arbitrary_rotation():
    source=trimesh.creation.box([.55,1.2,.28])
    source.apply_transform(trimesh.transformations.rotation_matrix(.63,[1,2,3]));source.apply_translation([2,-3,1])
    before=source.vertices.copy()
    physical,normalized,frame,record=prepare_appearance(source,profile(),target_length_mm=200)
    assert np.allclose(normalized.extents,[200*.55/1.2,200,200*.28/1.2],atol=.01)
    assert np.allclose(source.copy().apply_transform(frame['input_to_vehicle']).vertices,normalized.vertices)
    assert np.allclose(normalized.copy().apply_transform(frame['vehicle_to_input']).vertices,source.vertices)
    assert np.array_equal(source.vertices,before)
    assert record['basis']=='explicit_target_length' and not record['unit_was_guessed']


def test_declared_meters_preserve_real_dimensions_in_millimeters():
    _,normalized,frame,record=prepare_appearance(trimesh.creation.box([.055,.12,.028]),profile(),input_unit='m')
    assert np.allclose(normalized.extents,[55,120,28],atol=.01)
    assert record['source_coordinate_to_mm_factor']==1000
    assert np.isfinite(frame['front_signal_mm'])
    import json
    from smartcar.io import json_default
    json.dumps(frame,default=json_default,allow_nan=False)


@pytest.mark.parametrize('length',[0,-1,float('nan'),float('inf')])
def test_invalid_target_length_is_rejected(length):
    with pytest.raises(GeometryInputError,match='INVALID_TARGET_LENGTH'):
        prepare_appearance(trimesh.creation.box(),profile(),target_length_mm=length)


def test_empty_occupancy_never_reaches_surface_level_error():
    with pytest.raises(GeometryInputError,match='EMPTY_VOXEL_VOLUME'):
        occupancy_mesh(np.zeros((4,5,6),bool),np.zeros(3),.6)


def test_manufacturing_erosion_failure_retains_original_mesh():
    source=trimesh.creation.box([.5,1.1,.3]);before=source.vertices.copy()
    with pytest.raises(GeometryInputError,match='EMPTY_REPAIRED_VOLUME'):
        repair_volume(source,.6,1.6)
    assert np.array_equal(source.vertices,before)


def test_large_unit_conversion_hits_budget_before_voxel_allocation():
    with pytest.raises(GeometryInputError,match='VOXEL_GRID_LIMIT'):
        check_voxel_budget([550,1200,280],.6,80000000)
