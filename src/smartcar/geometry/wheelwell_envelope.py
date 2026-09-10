"""Local appearance-envelope reconstruction around measured obsolete wheels.

This is an explicit body redesign, not a claim that hardware fits the untouched
input. Only measured wheel regions can change. Body side widths come from the
surface immediately above each tire, and the belly comes from the central body.
The resulting envelope and changed volume are exported and independently used
for final shell containment/collision tests.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from smartcar.domain.vehicle_policy import VehiclePolicy
from smartcar.geometry.repair import occupancy_mesh


def reconstruct_wheel_regions(occupancy, origin, pitch, evidence, policy=None, width_strategy='crown_and_flanks'):
    policy = policy or VehiclePolicy()
    mask = np.asarray(occupancy, dtype=bool); result = mask.copy()
    origin = np.asarray(origin); axes = [origin[i]+np.arange(mask.shape[i])*pitch for i in range(3)]
    bounds = np.asarray(evidence['source_bounds_mm'])
    width, length, height = np.ptp(bounds, axis=0)
    middle = bounds.mean(0); zones=[]
    axles = evidence.get('axles', [])
    if len(axles) != 2:
        return result, dict(applied=False, reason='two measured wheel regions required', zones=[])
    central = abs(axes[0]-middle[0]) <= width*policy.belly_sample_width_fraction/2
    interior_y = (axes[1] > axles[0]['y_mm']+(axles[0].get('radius_mm') or 0)) & (axes[1] < axles[1]['y_mm']-(axles[1].get('radius_mm') or 0))
    field = mask[central][:,interior_y]
    bottoms = field.argmax(2)[field.any(2)]
    if not len(bottoms):
        return result, dict(applied=False, reason='no measured central belly surface', zones=[])
    belly = float(origin[2]+np.quantile(bottoms, policy.belly_height_quantile)*pitch)
    for axle in axles:
        radius= axle.get('radius_mm'); zc=axle.get('z_mm')
        if not radius or zc is None:
            continue
        y = axle['y_mm']; expanded = radius*(1+policy.wheelwell_radius_margin_fraction)
        # Use actual material above the old wheel crown. The paired outermost
        # coordinates are robust medians across several sections, not a global
        # bounding-box width or a supplied car-specific dimension.
        yy = abs(axes[1]-y) <= radius*0.35
        zz = (axes[2] >= zc+radius) & (axes[2] <= zc+expanded+2*pitch)
        samples = mask[:, yy][:,:,zz]
        occupied = samples.any(0)
        if not occupied.any():
            zones.append(dict(y_mm=y, applied=False, reason='wheel crown has no body reference'))
            continue
        first = samples.argmax(0)[occupied]
        last = samples.shape[0]-1-samples[::-1].argmax(0)[occupied]
        left = float(np.median(axes[0][first])); right = float(np.median(axes[0][last]))
        # A wheel-arch clearance can extend above the tire crown. Measure the
        # two neighbouring side panels too, so a notch is not mistaken for the
        # exterior body width and reconstructed as a narrow inward pedestal.
        flank_y=(abs(axes[1]-y)>=expanded)&(abs(axes[1]-y)<=expanded+radius*.35)
        flank_z=(axes[2]>=max(belly,zc))&(axes[2]<=zc+expanded)
        flank=mask[:,flank_y][:,:,flank_z];present=flank.any(0)
        if present.any() and width_strategy=='crown_and_flanks':
            flank_left=float(np.median(axes[0][flank.argmax(0)[present]]))
            flank_right=float(np.median(axes[0][flank.shape[0]-1-flank[::-1].argmax(0)[present]]))
            left=min(left,flank_left);right=max(right,flank_right)
        if right-left < width*0.35:
            zones.append(dict(y_mm=y, applied=False, reason='crown width is not a bilateral body section'))
            continue
        zone = (axes[1][:,None]-y)**2+(axes[2][None,:]-zc)**2 <= expanded**2
        # Filled material must meet the main belly. Below it the decorative
        # wheel is removed; above the measured crown the original mesh remains.
        shape = ((axes[0]>=left)&(axes[0]<=right))[:,None,None] & (axes[2]>=belly)[None,None,:]
        before = int(result[:,zone].sum())
        replacement = np.broadcast_to(shape, mask.shape)
        result[:,zone] = replacement[:,zone]
        zones.append(dict(applied=True, y_mm=y, source_radius_mm=radius, source_z_mm=zc,
                          redesign_radius_mm=expanded, reference_body_x_mm=[left,right], belly_z_mm=belly,
                          occupied_before=before, occupied_after=int(result[:,zone].sum())))
    added = result & ~mask; removed = mask & ~result
    fraction = float(added.sum()/max(1,mask.sum()))
    # Preserve all geometry outside the measured masks, including detached
    # source details; connectivity is checked later on the actual print parts.
    return result, dict(applied=bool(np.any(added)|np.any(removed)), method='measured wheel-region substitution with crown sidewalls and central belly',width_strategy=width_strategy,
                        zones=zones, added_mm3=float(added.sum()*pitch**3), removed_mm3=float(removed.sum()*pitch**3),
                        added_fraction=fraction, maximum_addition_fraction=policy.maximum_wheelwell_addition_fraction,
                        status='FAIL' if fraction>policy.maximum_wheelwell_addition_fraction else 'WARNING',
                        warning='Local wheel wells are redesigned for fixed real tires; exterior change is measured and requires appearance review.')
