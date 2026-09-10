"""Additional frame evidence for fused tires missed by component recognition."""
import numpy as np
from smartcar.understanding.axle_anchors import detect_axle_anchors


def rescore_ground_frames(mesh, candidates,policy=None):
    for candidate in candidates:
        rotation=np.asarray(candidate['rotation'])
        vertices=mesh.vertices@rotation.T
        spans=np.ptp(vertices,axis=0)
        # A wheel-axle frame has vehicle length along its longest candidate
        # direction. Other permutations retain the original generic evidence.
        if spans[1]<spans.max()*.9:continue
        transformed=mesh.copy()
        transformed.vertices=vertices
        evidence=detect_axle_anchors(transformed,tolerance=spans.max()*.003,policy=policy)
        candidate['ground_contact_evidence']=evidence
        if len(evidence['axles'])!=2:continue
        # Require plausible lower circular sections as well as two contact
        # regions. Roof decorations must not outweigh underbody evidence.
        axles=evidence['axles'];radii=[a.get('radius_mm') for a in axles]
        if not all(radii):continue
        grounded=all(abs(a['z_mm']-a['radius_mm']-vertices[:,2].min())<=max(spans.max()*.025,a['radius_mm']*.25) for a in axles)
        bonus=1.5 if grounded else 0.
        candidate['score_components']['bilateral_ground_contact']=bonus
        candidate['score']=float(sum(candidate['score_components'].values()))
    return candidates
