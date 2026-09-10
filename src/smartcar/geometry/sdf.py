from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, minimum_filter1d
from smartcar.geometry.repair import occupancy_mesh


def resample_designable_volume(mesh,pitch,wall,max_cells=80000000):
    """Recompute geometry queries in physical mm after appearance scaling.

    Merely scaling an existing grid also scales its distance uncertainty and can
    erase an otherwise valid wheel ride-height interval. Revoxelize the actual
    current exterior; do not resize hardware or reduce the required clearance.
    """
    from smartcar.geometry.voxelize import voxelize_closed_solid
    mask,origin,record=voxelize_closed_solid(mesh,pitch,max_cells=max_cells)
    volume=DesignableVolume(mask,origin,pitch,wall)
    record.update(pitch_mm=pitch,distance_uncertainty_mm=volume.error,wall_mm=wall)
    return volume,record


class DesignableVolume:
    def __init__(self, occupancy, origin, pitch, wall):
        self.occupancy = occupancy
        self.origin = np.asarray(origin)
        self.pitch = pitch
        self.wall = wall
        # EDT is center distance. Subtract a full half voxel diagonal for a
        # conservative lower bound to the reconstructed exterior surface.
        self.error = .5*np.sqrt(3)*pitch
        self.distance = distance_transform_edt(occupancy).astype(np.float32)*pitch-self.error
        self.distance[~occupancy] = -distance_transform_edt(~occupancy)[~occupancy]*pitch
        self.allowed = self.distance >= wall

    def query(self, points):
        p = (np.asarray(points)-self.origin)/self.pitch
        return map_coordinates(self.distance, p.T, order=1, mode="constant", cval=-1e6)

    def box_samples(self, bounds, step=None):
        b = np.asarray(bounds)
        step = step or self.pitch*2
        axes = [np.linspace(b[0,i],b[1,i],max(2,int(np.ceil((b[1,i]-b[0,i])/step))+1)) for i in range(3)]
        # All box interior points, not only corners: concave exterior can cut a face.
        return np.stack(np.meshgrid(*axes,indexing="ij"),-1).reshape(-1,3)

    def box_clearance(self, bounds, step=None):
        return float(self.query(self.box_samples(bounds,step)).min())

    def mesh(self): return occupancy_mesh(self.allowed,self.origin,self.pitch)

    def feasible_centers(self, extents, clearance, bottom=None, stride=1,diagnostics=None):
        # Separable rectangular erosion gives a conservative FULL-volume test.
        field=self.distance.copy()
        for axis,size in enumerate(extents):
            cells=int(np.ceil((size/2+clearance+self.error)/self.pitch))
            field=minimum_filter1d(field,size=2*cells+1,axis=axis,mode="constant",cval=-1e6)
        valid=field>=self.wall
        if diagnostics is not None:diagnostics['inside_centers_before_floor']=int(valid.sum())
        if bottom is not None:
            valid[:,:,self.origin[2]+np.arange(valid.shape[2])*self.pitch < bottom+extents[2]/2]=False
        if diagnostics is not None:diagnostics['inside_centers_after_floor']=int(valid.sum())
        stride=np.broadcast_to(np.asarray(stride),(3,)).astype(int)
        idx=np.argwhere(valid[::stride[0],::stride[1],::stride[2]])*stride
        if diagnostics is not None:diagnostics['sampled_centers']=len(idx)
        return self.origin+idx*self.pitch,field[tuple(idx.T)] if len(idx) else np.array([])
