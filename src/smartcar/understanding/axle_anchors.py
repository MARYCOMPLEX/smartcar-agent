"""Recover axle regions from the current appearance, without retained poses.

Separate circular components provide direct evidence. Horizontal sections near
ground provide a second measurement for fused, rough and non-circular tires.
The section method requires two separated bilateral contact bands which agree
at several heights. A flat box has no such evidence and gets an explicit fallback.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from smartcar.domain.vehicle_policy import VehiclePolicy


def horizontal_segments(mesh, height):
    triangles = mesh.triangles
    z = triangles[:, :, 2]
    active = (z.min(1) < height) & (z.max(1) > height)
    triangles = triangles[active]
    if not len(triangles):
        return np.empty((0, 2, 3))
    points = np.zeros((len(triangles), 2, 3))
    counts = np.zeros(len(triangles), dtype=int)
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        a, b = triangles[:, i], triangles[:, j]
        crossing = ((a[:, 2] <= height) & (b[:, 2] > height)) | ((b[:, 2] <= height) & (a[:, 2] > height))
        indices = np.flatnonzero(crossing)
        t = (height-a[indices, 2])/(b[indices, 2]-a[indices, 2])
        points[indices, counts[indices]] = a[indices]+t[:, None]*(b[indices]-a[indices])
        counts[indices] += 1
    return points[counts == 2]


def _side_intervals(segments, threshold, sign):
    if not len(segments):
        return np.empty((0, 2))
    a, b = segments[:, 0].copy(), segments[:, 1].copy()
    va, vb = sign*a[:, 0]-threshold, sign*b[:, 0]-threshold
    keep = (va >= 0) | (vb >= 0)
    a, b, va, vb = a[keep], b[keep], va[keep], vb[keep]
    for values, other, d, other_d in [(a, b, va, vb), (b, a, vb, va)]:
        outside = d < 0
        t = -d[outside]/(other_d[outside]-d[outside])
        values[outside] += t[:, None]*(other[outside]-values[outside])
    return np.sort(np.stack([a[:, 1], b[:, 1]], axis=1), axis=1)


def _contact_sections(mesh, policy):
    bounds = mesh.bounds; length = mesh.extents[1]
    n = policy.contact_bins; dy = length/n
    sections = []
    for fraction in policy.contact_section_height_fractions:
        height = bounds[0, 2]+fraction*mesh.extents[2]
        segments = horizontal_segments(mesh, height)
        # Work around the actual bilateral midplane, not source-origin X=0.
        segments = segments.copy(); segments[:, :, 0] -= bounds.mean(0)[0]
        masks = []
        for side in [1, -1]:
            intervals = _side_intervals(segments, policy.contact_lateral_fraction*mesh.extents[0], side)
            mask = np.zeros(n, dtype=bool)
            for a, b in intervals:
                lo = max(0, int(np.floor((a-bounds[0, 1])/dy)))
                hi = min(n, int(np.ceil((b-bounds[0, 1])/dy))+1)
                mask[lo:hi] = True
            masks.append(mask)
        both = masks[0] & masks[1]
        both = ndi.binary_closing(np.pad(both, 1), iterations=max(1, int(policy.contact_gap_fraction*n)))[1:-1]
        labels, count = ndi.label(both)
        bands = []
        for label in range(1, count+1):
            indices = np.flatnonzero(labels == label)
            width = len(indices)/n
            if policy.minimum_contact_band_fraction <= width <= policy.maximum_contact_band_fraction:
                lo, hi = bounds[0, 1]+np.array([indices[0], indices[-1]+1])*dy
                bands.append(dict(y_interval_mm=[float(lo), float(hi)], y_mm=float((lo+hi)/2), height_mm=float(height)))
        record = dict(height_mm=float(height), bands=bands, accepted=False)
        if len(bands) == 2:
            separation = bands[1]['y_mm']-bands[0]['y_mm']
            record['accepted'] = bool(policy.minimum_wheelbase_fraction*length <= separation <= policy.maximum_wheelbase_fraction*length)
        sections.append(record)
    return sections


def detect_axle_anchors(mesh, semantic=None, tolerance=0.6, policy=None):
    policy = policy or VehiclePolicy()
    semantic = semantic or {}
    length = float(mesh.extents[1]); groups = []
    for evidence in sorted(semantic.get('removed_components', []), key=lambda r: r['center'][1]):
        y = float(evidence['center'][1])
        if not groups or abs(y-np.mean([e['center'][1] for e in groups[-1]])) > max(2*tolerance, length/policy.contact_bins):
            groups.append([])
        groups[-1].append(evidence)
    direct = []
    for group in groups:
        if sum(e['wheel_count'] for e in group) < 2:
            continue
        direct.append(dict(y_mm=float(np.mean([e['center'][1] for e in group])),
                           z_mm=float(np.mean([e['center'][2] for e in group])),
                           radius_mm=float(np.median([e['radius_mm'] for e in group])),
                           evidence='bilateral_circular_components', components=[e['component'] for e in group]))
    sections = _contact_sections(mesh, policy)
    accepted = [s for s in sections if s['accepted']]
    agreed = []
    if accepted:
        for seed in accepted:
            centers = np.array([b['y_mm'] for b in seed['bands']])
            neighbours = [s for s in accepted if np.max(np.abs(np.array([b['y_mm'] for b in s['bands']])-centers)) <= policy.contact_center_agreement_fraction*length]
            if len(neighbours) > len(agreed):
                agreed = neighbours
    measured = []
    if len(agreed) >= policy.minimum_agreeing_sections:
        for index in range(2):
            bands = [s['bands'][index] for s in agreed]
            y = float(np.median([b['y_mm'] for b in bands]))
            # Fit the lower circular silhouette from independent section edges.
            # Radius is advisory; axle Y is obtained directly from contact bands.
            points = np.array([[edge, b['height_mm']] for b in bands for edge in b['y_interval_mm']])
            matrix = np.column_stack([2*points[:, 0], 2*points[:, 1], np.ones(len(points))])
            fit, *_ = np.linalg.lstsq(matrix, (points**2).sum(1), rcond=None)
            radius = float(np.sqrt(max(0, fit[2]+fit[0]**2+fit[1]**2)))
            residual = float(np.sqrt(np.mean((np.linalg.norm(points-fit[:2], axis=1)-radius)**2)))
            plausible = tolerance < radius < policy.maximum_contact_band_fraction*length/2 and abs(fit[0]-y) < policy.contact_center_agreement_fraction*length
            measured.append(dict(y_mm=y, z_mm=float(fit[1]) if plausible else None,
                                 radius_mm=radius if plausible else None,
                                 radius_fit_rms_mm=residual, evidence='bilateral_ground_sections',
                                 agreeing_heights_mm=[s['height_mm'] for s in agreed],
                                 center_spread_mm=float(np.ptp([b['y_mm'] for b in bands]))))
    if len(direct) == 2:
        axles = direct; method = 'two circular component axle pairs'
    elif len(measured) == 2:
        axles = measured; method = 'two bilateral ground-contact regions'
        for axle in axles:
            nearest = min(direct, key=lambda d: abs(d['y_mm']-axle['y_mm'])) if direct else None
            if nearest and abs(nearest['y_mm']-axle['y_mm']) < policy.source_axle_tolerance_fraction*length:
                axle.update(nearest)
    else:
        axles = []; method = 'shape-based wheelbase policy; source axle evidence unresolved'
    return dict(schema_version='axle_evidence.v1', method=method,
                source_bounds_mm=mesh.bounds.tolist(), source_dimensions_mm=mesh.extents.tolist(),
                axles=sorted(axles, key=lambda a: a['y_mm']), source_wheelbase_mm=float(axles[1]['y_mm']-axles[0]['y_mm']) if len(axles)==2 else None,
                direct_component_axles=direct, contact_sections=sections,
                confidence='measured' if len(axles)==2 else 'fallback',
                warning='Front sign remains semantic; axle correspondence is invariant to swapping front and rear.')


def scaled_axle_targets(evidence, scale, bounds, tire_radius, profile, policy=None):
    policy = policy or VehiclePolicy(); bounds = np.asarray(bounds)
    length = float(np.ptp(bounds, axis=0)[1])
    axles = evidence.get('axles', [])
    measured = len(axles) == 2
    targets = [a['y_mm']*scale for a in axles] if measured else list(bounds.mean(0)[1]+np.array([-1, 1])*length*policy.preferred_wheelbase_fraction/2)
    source_wheelbase = targets[1]-targets[0]
    return dict(measured=measured, target_y_mm=targets, bounds_mm=bounds.tolist(),
                tolerance_mm=max(profile.candidate_pitch, length*policy.source_axle_tolerance_fraction),
                minimum_wheelbase_mm=max(length*policy.minimum_wheelbase_fraction,
                    2*tire_radius*(1+policy.minimum_tire_gap_diameters),
                    source_wheelbase*policy.minimum_source_wheelbase_retention if measured else 0),
                maximum_wheelbase_mm=length*policy.maximum_wheelbase_fraction,
                maximum_overhang_imbalance_mm=length*policy.maximum_overhang_imbalance_fraction,
                source_wheel_radii_mm=[a['radius_mm']*scale if a.get('radius_mm') else None for a in axles],
                policy=policy.to_dict())
