"""Measured lateral body reference around an axle, including its fender crown."""
import numpy as np


def axle_body_halfwidth(volume, center, radius, bounds):
    axes=[volume.origin[i]+np.arange(volume.occupancy.shape[i])*volume.pitch for i in range(3)]
    # The empty middle of a source arch is not the body side. Use neighbouring
    # Y sections and the full upper wheel disk, robust to small roof ornaments.
    y=abs(axes[1]-center[1])<=radius*1.35
    z=(axes[2]>=center[2])&(axes[2]<=center[2]+radius+volume.wall)
    region=volume.occupancy[:,y][:,:,z];present=region.any(0)
    if present.any():
        first=region.argmax(0)[present]
        last=region.shape[0]-1-region[::-1].argmax(0)[present]
        extents=np.maximum(abs(axes[0][first]),abs(axes[0][last]))
        return float(np.quantile(extents,.9))
    return float(np.ptp(np.asarray(bounds),axis=0)[0]/2)
