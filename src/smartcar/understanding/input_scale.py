"""Explicit source-unit or length calibration; never guess an STL's units."""
import numpy as np
from smartcar.geometry.errors import GeometryInputError
from smartcar.understanding.coordinate_frame import establish_frame

UNIT_TO_MM={'mm':1.,'cm':10.,'m':1000.,'in':25.4}


def prepare_appearance(source,profile,input_unit=None,target_length_mm=None,policy=None):
    extents=np.asarray(source.extents,dtype=float)
    if not np.isfinite(source.vertices).all() or np.max(extents)<=0:
        raise GeometryInputError('INVALID_APPEARANCE_BOUNDS','Input needs finite nonzero geometry.')
    if input_unit is not None and target_length_mm is not None:
        raise GeometryInputError('AMBIGUOUS_INPUT_SCALE','Specify either --input-unit or --target-length-mm.')
    if target_length_mm is not None:
        if not np.isfinite(target_length_mm) or target_length_mm<=0:
            raise GeometryInputError('INVALID_TARGET_LENGTH','--target-length-mm must be finite and positive.')
        # Condition the frame calculation at physical scale first. After finding
        # the frame, calibrate its actual Y length, not a rotated source AABB.
        initial_factor=target_length_mm/float(extents.max());basis='explicit_target_length'
    else:
        initial_factor=UNIT_TO_MM[input_unit or 'mm'];basis='explicit_unit' if input_unit else 'assumed_mm'
    physical=source.copy().apply_scale(initial_factor)
    if physical.extents.max()*profile.maximum_scale<=2*profile.nominal_wall:
        raise GeometryInputError('INPUT_SCALE_TOO_SMALL',
            'Model is smaller than the configured shell walls even at maximum design scale. STL has no reliable units; specify --input-unit mm|cm|m|in or --target-length-mm <starting body length>.',
            source_dimensions=extents,assumed_input_unit=input_unit or 'mm',dimensions_mm=physical.extents,
            voxel_pitch_mm=profile.voxel_pitch,minimum_wall_mm=profile.minimum_wall,
            maximum_design_scale=profile.maximum_scale,original_file_modified=False)
    normalized,frame=establish_frame(physical,policy)
    correction=1. if target_length_mm is None else float(target_length_mm/normalized.extents[1])
    if correction!=1.:
        normalized.apply_scale(correction);physical.apply_scale(correction)
        frame['front_signal_mm']*=correction
        frame['symmetry']={k:v*correction for k,v in frame['symmetry'].items()}
        for candidate in frame['candidates']:
            candidate['low_flat_area_mm2']*=correction**2
            if 'bilateral_symmetry' in candidate:
                candidate['bilateral_symmetry']={k:v*correction for k,v in candidate['bilateral_symmetry'].items()}
            for wheel in candidate.get('wheel_evidence',[]):
                wheel['center']=np.asarray(wheel['center'])*correction
                for key in ['radius_mm','width_mm','ground_gap_mm']:wheel[key]*=correction
    factor=initial_factor*correction
    rigid=np.array(frame['input_to_vehicle'],copy=True);rigid[:3,3]*=correction
    calibration=np.diag([factor,factor,factor,1.])
    frame['input_mm_to_vehicle_mm']=rigid
    frame['input_to_vehicle']=rigid@calibration
    frame['vehicle_to_input']=np.linalg.inv(frame['input_to_vehicle'])
    frame['dimensions_mm']=normalized.extents
    frame['source_coordinate_to_mm_factor']=factor
    record=dict(basis=basis,input_unit=input_unit or ('unspecified' if target_length_mm is not None else 'mm'),
                source_dimensions=extents,source_coordinate_to_mm_factor=factor,target_length_mm=target_length_mm,
                prepared_dimensions_mm=normalized.extents,
                unit_was_guessed=False,original_file_modified=False,
                downstream_design_scale='separate bounded layout scale search; may enlarge this starting appearance')
    return physical,normalized,frame,record
