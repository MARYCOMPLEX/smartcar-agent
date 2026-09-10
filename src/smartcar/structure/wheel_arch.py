import numpy as np
from smartcar.geometry.solid import cylinder,box
from smartcar.geometry.wheel import arch_radius


def wheel_arch(center,radius,width,bottom,profile,extra=0.,exterior_bounds=None):
    """Open-bottom arch: circular crown, vertical sides below axle height.

    Extending the opening to the underside avoids a circular Boolean grazing
    a horizontal bottom plate and creating unprintable feather edges.
    """
    c=np.asarray(center)
    r=arch_radius(radius,profile,extra)
    half=width/2+profile.moving_clearance+extra
    xlo,xhi=c[0]-half,c[0]+half
    if exterior_bounds is not None:
        # A tire may sit in a shallow side recess. Carry its generated aperture
        # to the exterior so no outer membrane blocks axial assembly or access.
        if c[0]>0:xhi=max(xhi,float(exterior_bounds[1,0])+profile.moving_clearance+extra)
        else:xlo=min(xlo,float(exterior_bounds[0,0])-profile.moving_clearance-extra)
    arc=cylinder(r,xlo,xhi,c[1:],axis=0)
    skirt=box([[xlo,c[1]-r,bottom],[xhi,c[1]+r,c[2]]])
    return arc+skirt
