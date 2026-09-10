"""Remove sub-profile bottom edge webs after actual wheel/access cutting."""
import numpy as np
from smartcar.geometry.solid import box,to_mesh


def regularize_bottom(chassis,floor,profile):
    section=chassis.slice(floor-profile.bottom_thickness/2)
    r=profile.nominal_wall/2
    opened=section.offset(-r).offset(r)^section
    bounds=to_mesh(chassis).bounds.copy();bounds[0,:2]-=1;bounds[1,:2]+=1
    bounds[0,2]=min(bounds[0,2]-1,floor-profile.bottom_thickness-1);bounds[1,2]=floor
    upper=chassis-box(bounds)
    slab=opened.extrude(profile.bottom_thickness).translate((0,0,floor-profile.bottom_thickness))
    return upper+slab,dict(method='planar rolling disk after wheel and access cuts, clipped to original section',
        rolling_diameter_mm=2*r,section_before_mm2=section.area(),section_after_mm2=opened.area(),
        removed_base_mm3=(section.area()-opened.area())*profile.bottom_thickness)
