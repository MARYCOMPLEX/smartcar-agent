"""Measured evidence for separate wheels and connected wheel/axle assemblies."""
import numpy as np


def wheel_component_evidence(components, bounds, tolerance, rotation=None):
    bounds=np.asarray(bounds);span=np.ptp(bounds,axis=0);middle=bounds.mean(0)
    candidates=[]
    for index,part in enumerate(components):
        vertices=part.vertices if rotation is None else part.vertices@rotation.T
        lo=vertices.min(0);hi=vertices.max(0);d=hi-lo;c=(hi+lo)/2
        if min(d[1:])<tolerance*4 or max(d[1:])/min(d[1:])>1.08:continue
        radius=max(d[1:])/2
        radial=np.linalg.norm(vertices[:,1:]-c[1:],axis=1)
        radial_error=float(np.quantile(radial,.98)/radius)
        if radial_error>1.04 or hi[2]>bounds[0,2]+.65*span[2]:continue
        # A connected pair can be wider than its diameter. Unlike a single
        # lateral wheel, it spans the bilateral midplane and both side regions.
        axle=(abs(c[0]-middle[0])<2*tolerance and
              lo[0]<middle[0]-.3*span[0] and hi[0]>middle[0]+.3*span[0])
        single=d[0]<=.6*min(d[1:]) and abs(c[0]-middle[0])>=.3*span[0]
        if not (axle or single):continue
        candidates.append(dict(component=index,center=c,radius_mm=radius,width_mm=d[0],
                               type='connected_axle_pair' if axle else 'single_wheel',
                               wheel_count=2 if axle else 1,radial_quantile_ratio=radial_error,
                               ground_gap_mm=float(lo[2]-bounds[0,2])))
    accepted=[]
    for a in candidates:
        if a['type']=='connected_axle_pair':accepted.append(a);continue
        reflected=a['center'].copy();reflected[0]=2*middle[0]-reflected[0]
        if any(b['component']!=a['component'] and b['type']=='single_wheel' and
               np.linalg.norm(b['center']-reflected)<2*tolerance and
               abs(b['radius_mm']-a['radius_mm'])<tolerance for b in candidates):accepted.append(a)
    return accepted
