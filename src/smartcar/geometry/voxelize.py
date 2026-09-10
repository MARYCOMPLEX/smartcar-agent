"""Bounded-memory scanline occupancy for closed engineering solids."""
import numpy as np
from smartcar.geometry.rays import all_ray_hits


def voxelize_closed_solid(mesh,pitch,batch_size=256,max_cells=80000000):
    from smartcar.geometry.repair import check_voxel_budget
    check_voxel_budget(mesh.extents,pitch,max_cells)
    origin=mesh.bounds[0]-1.5*pitch
    shape=np.ceil((mesh.bounds[1]-origin)/pitch).astype(int)+3
    axis=int(np.argmax(shape));other=[i for i in range(3) if i!=axis]
    indices=np.stack(np.meshgrid(np.arange(shape[other[0]]),np.arange(shape[other[1]]),indexing='ij'),-1).reshape(-1,2)
    mask=np.zeros(shape,dtype=bool);ambiguous=0
    for start in range(0,len(indices),batch_size):
        ij=indices[start:start+batch_size];origins=np.tile(origin,(len(ij),1))
        # Irrational-ratio perturbations avoid rays through shared voxel edges.
        origins[:,other]=origin[other]+ij*pitch+pitch*1e-5*np.array([1.,.61803398875])
        origins[:,axis]=origin[axis]-pitch
        directions=np.zeros_like(origins);directions[:,axis]=1
        points,rays,_=all_ray_hits(mesh,origins,directions)
        if not len(points):continue
        order=np.lexsort((points[:,axis],rays));points=points[order];rays=rays[order]
        ray_ids,starts,counts=np.unique(rays,return_index=True,return_counts=True)
        for ray,index,count in zip(ray_ids,starts,counts):
            coordinates=points[index:index+count,axis]
            if count%2:
                ambiguous+=1
                # Evaluate the exceptional whole column using an independent
                # direction chosen by Trimesh's containment routine.
                samples=np.tile(origins[ray],(shape[axis],1));samples[:,axis]=origin[axis]+np.arange(shape[axis])*pitch
                occupancy=mesh.contains(samples)
            else:
                occupancy=np.zeros(shape[axis],bool)
                for low,high in coordinates.reshape(-1,2):
                    lo=max(0,int(np.ceil((low-origin[axis])/pitch)))
                    hi=min(shape[axis],int(np.floor((high-origin[axis])/pitch))+1)
                    occupancy[lo:hi]=True
            address=[None,None,None];address[axis]=slice(None)
            address[other[0]]=ij[ray,0];address[other[1]]=ij[ray,1]
            mask[tuple(address)]=occupancy
    return mask,origin,dict(method='batched closed-solid scanline parity',ray_axis=axis,columns=len(indices),batch_size=batch_size,
                            exceptional_columns=ambiguous,grid_shape=shape,grid_cells=int(np.prod(shape)))
