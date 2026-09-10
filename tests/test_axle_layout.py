import numpy as np
from pathlib import Path
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.validation.axle_layout import validate_axle_positions

PROFILE=ManufacturingProfile.load(Path(__file__).resolve().parents[1]/'config/manufacturing.json')


def positions(y):
    return [[x,yy,18.5] for yy in y for x in [-42,42]]


def test_collision_free_clustered_wheels_are_rejected():
    # 38 mm spacing clears two 37 mm tires, but is unacceptable for a 240 mm
    # car whose actual source axles are 145 mm apart.
    checks=validate_axle_positions(positions([-20,18]),[18.5]*4,[[-50,-120,0],[50,120,70]],[{'y_mm':-73},{'y_mm':72}],1,PROFILE)
    failed={c['check'] for c in checks if c['status']=='FAIL'}
    assert 'axle_layout:wheelbase' in failed
    assert 'axle_layout:source_correspondence' in failed


def test_spread_out_but_wrong_arches_are_also_rejected():
    checks=validate_axle_positions(positions([-56,90]),[18.5]*4,[[-50,-120,0],[50,120,70]],[{'y_mm':-73},{'y_mm':72}],1,PROFILE)
    assert next(c for c in checks if c['check']=='axle_layout:wheelbase')['status']=='PASS'
    assert next(c for c in checks if c['check']=='axle_layout:source_correspondence')['status']=='FAIL'


def test_measured_pose_matches_source_and_spacing():
    checks=validate_axle_positions(positions([-72,73]),[18.5]*4,[[-50,-120,0],[50,120,70]],[{'y_mm':-73},{'y_mm':72}],1,PROFILE)
    assert all(c['status']=='PASS' for c in checks)
