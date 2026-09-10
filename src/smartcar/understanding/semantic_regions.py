import numpy as np
import trimesh
from smartcar.understanding.wheel_evidence import wheel_component_evidence


def separate_decorative_wheels(mesh,tolerance):
    """Identify paired lateral circular components from measured geometry.
    Only distinct components with bilateral counterparts may be removed. Fused
    wheels need a stronger segmentation method and remain explicitly unresolved.
    """
    components=mesh.split(only_watertight=False)
    removed=wheel_component_evidence(components,mesh.bounds,tolerance)
    removed_ids={c["component"] for c in removed}
    kept=[c for i,c in enumerate(components) if i not in removed_ids]
    # Preserve the appearance when no body component can be distinguished.
    if not kept:return mesh.copy(),dict(method='circular component evidence',removed_components=[],retained_components=len(components),warning='all components resemble wheels; body classification unresolved')
    return trimesh.util.concatenate(kept),dict(method="bilateral circular wheels or connected wheel/axle pairs",removed_components=removed,
                                               retained_components=len(kept),warning="fused decorative wheels are not yet segmented")
