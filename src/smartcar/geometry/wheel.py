import numpy as np


def wheel_face_points(center,radius,width,side,samples=256):
    center=np.asarray(center);n=np.arange(samples)+.5
    radii=radius*np.sqrt(n/samples);angles=n*np.pi*(3-np.sqrt(5))
    points=np.tile(center,(samples,1)).astype(float)
    points[:,0]+=side*width/2
    points[:,1]+=radii*np.cos(angles);points[:,2]+=radii*np.sin(angles)
    return points


def wheel_side_exposure(volume,center,radius,width,side,samples=256,maximum_recess=0.):
    """Measure local exposure and feasibility of a shallow wheel-face aperture.

    A global vehicle width includes roof/fender flares and cannot determine
    whether a tire at a different Y/Z is exposed. An optional bounded recess
    reserves an explicit outward arch opening. It cannot accept deeply buried
    wheels; final axial installation and exterior access are independently
    validated after generating that opening. Uncertain samples are not credited.
    """
    points=wheel_face_points(center,radius,width,side,samples)
    distances=volume.query(points)
    targets=points.copy();targets[:,0]+=side*maximum_recess
    aperture_distances=volume.query(targets) if maximum_recess else distances
    return dict(exposed_fraction=float(np.mean(aperture_distances<=-volume.error)),
                initially_exposed_fraction=float(np.mean(distances<=-volume.error)),samples=samples,
                maximum_face_recess_mm=maximum_recess,distance_uncertainty_mm=volume.error,
                method='area-uniform wheel-face rays reach exterior within the declared shallow aperture depth')


def arch_radius(radius,profile,extra=0.):
    """Circumscribed 64-segment motion clearance envelope."""
    return (radius+profile.moving_clearance+profile.mesh_tolerance+extra)/np.cos(np.pi/64)


def wheel_end_margin(radius,profile):
    # Include the shell reconstruction reserve and inset chassis perimeter.
    # End webs are synthesized at nominal wall thickness, never at an already
    # tolerance-consumed minimum before the actual wheel arch is subtracted.
    uncertainty=.5*np.sqrt(3)*profile.shell_regularization_pitch
    return arch_radius(radius,profile,uncertainty)-radius+profile.nominal_wall+profile.sliding_clearance
