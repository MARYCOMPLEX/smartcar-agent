from smartcar.geometry.solid import box,cylinder,to_mesh


def generate_fit_coupon(profile):
    """Small print to check supplied fastener calibration on the target machine.
    It is a separate test artifact and never part of the car assembly.
    """
    diameters=[profile.screw_fit["horizontal_pilot"],profile.screw_fit["vertical_pilot"],profile.screw_fit["clearance_hole"]]
    spacing=max(diameters)+2*profile.nominal_wall
    height=profile.screw_fit["thread_length"]+profile.bottom_thickness
    coupon=box([[0,0,0],[spacing*len(diameters),spacing,height]])
    positions=[]
    for i,d in enumerate(diameters):
        x=(i+.5)*spacing;y=spacing/2
        coupon=coupon-cylinder(d/2,profile.bottom_thickness,height+.1,(x,y))
        positions.append(dict(index=i+1,center_xy_mm=[x,y],mode="vertical print test",nominal_diameter_mm=d))
    return coupon,dict(holes=positions,warning="The horizontal-pilot diameter is included as a size comparison only; this coupon does not calibrate horizontal-hole orientation.",height_mm=height)
