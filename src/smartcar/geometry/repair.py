from __future__ import annotations
import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from smartcar.geometry.errors import GeometryInputError


def check_voxel_budget(extents,pitch,max_cells=80000000):
    import math
    if not np.isfinite(pitch) or pitch<=0:raise GeometryInputError('INVALID_VOXEL_PITCH','Voxel pitch must be finite and positive.')
    shape=[int(math.ceil(float(v)/pitch))+12 for v in extents]
    cells=math.prod(shape)
    if cells>max_cells:
        raise GeometryInputError('VOXEL_GRID_LIMIT',
            'Requested model/resolution exceeds the configured grid budget. Check units or --target-length-mm before changing manufacturing resolution.',
            estimated_grid_shape=shape,estimated_cells=cells,maximum_cells=max_cells,voxel_pitch_mm=pitch)
    return dict(estimated_grid_shape=shape,estimated_cells=cells,maximum_cells=max_cells)


def occupancy_mesh(mask, origin, pitch, smoothing_radius_mm=0.):
    if np.ndim(mask)!=3 or not np.size(mask) or not np.any(mask):
        raise GeometryInputError('EMPTY_VOXEL_VOLUME','No occupied cells remain; check input dimensions, repair and minimum feature size.',grid_shape=list(np.shape(mask)))
    # Padding makes every extracted surface closed, even when touching grid edges.
    padded = np.pad(mask.astype(np.float32),1)
    if smoothing_radius_mm>0:
        # Reconstruct material at a specified physical scale, avoiding the
        # single-cell diagonal spikes of a binary marching surface. This is
        # a geometry operation, never a wall-thickness acceptance allowance.
        padded=ndi.gaussian_filter(padded,sigma=smoothing_radius_mm/pitch,mode='constant',cval=0.)
        if padded.max()<=.5:
            raise GeometryInputError('EMPTY_SMOOTHED_VOLUME','No material survives the configured reconstruction scale.',smoothing_radius_mm=smoothing_radius_mm)
    # Binary occupancy has exact integer/half-integer contour coordinates.
    # Keep those exact lattice coordinates until conversion to float64. Applying
    # physical spacing or a translated origin in skimage's float32 array creates
    # micrometre plane wobble, which becomes degenerate faces at STL export.
    verts,faces,_,_ = marching_cubes(padded, .5, spacing=(1.,)*3, allow_degenerate=False)
    verts = np.asarray(verts,dtype=np.float64)*pitch + np.asarray(origin,dtype=np.float64) - pitch
    m = trimesh.Trimesh(verts, faces, process=True)
    trimesh.repair.fix_normals(m, multibody=True)
    return m


def rolling_sphere_mesh(mask, origin, pitch, diameter):
    """Reconstruct a union of interior balls without blurring voids into webs.

    Each retained center has the requested material radius in the sampled
    occupancy. Account for the half-cell boundary when expanding the centers.
    The resulting distance field describes balls, not a smoothed binary label:
    Gaussian labels can join nearby walls by a sub-nozzle bridge. The final mesh
    still needs independent wall, clearance, topology and assembly checks.
    """
    radius=diameter/2
    if radius<=pitch/2:
        raise GeometryInputError('ROLLING_RADIUS_BELOW_GRID','Rolling sphere must exceed half a grid cell.')
    distance=ndi.distance_transform_edt(mask,sampling=pitch)
    centers=distance>=radius-1e-9
    del distance
    if not centers.any():
        raise GeometryInputError('EMPTY_ROLLING_VOLUME','No interior ball fits the configured wall reconstruction.')
    field=ndi.distance_transform_edt(~centers,sampling=pitch)
    field *= -1
    field += radius-pitch/2
    verts,faces,_,_=marching_cubes(field,0.,allow_degenerate=False)
    mesh=trimesh.Trimesh(np.asarray(verts,dtype=np.float64)*pitch+np.asarray(origin),faces,process=True)
    trimesh.repair.fix_normals(mesh,multibody=True)
    return mesh


def repair_volume(mesh, pitch, minimum_feature=0.,max_cells=80000000):
    budget=check_voxel_budget(mesh.extents,pitch,max_cells)
    m = mesh.copy()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m, multibody=True)
    trimesh.repair.fill_holes(m)
    vox = m.voxelized(pitch)
    surf = np.pad(vox.matrix,4)
    origin = vox.transform[:3,3] - 4*pitch
    closed = ndi.binary_closing(surf, iterations=2)
    filled = ndi.binary_fill_holes(closed)
    labels,n = ndi.label(filled)
    if n==0:raise GeometryInputError('EMPTY_REPAIRED_VOLUME','Voxel closing/fill produced no foreground volume.',voxel_pitch_mm=pitch)
    counts = np.bincount(labels.ravel()); counts[0] = 0
    main = labels == counts.argmax()
    # Fill vertical columns enclosed by exterior samples: handles open pickup beds
    # and imperfect shells. This is an appearance envelope, not hardware clearance.
    occupied = main.any(2)
    first = main.argmax(2)
    last = main.shape[2]-1-main[:,:,::-1].argmax(2)
    z = np.arange(main.shape[2])[None,None,:]
    envelope = occupied[:,:,None] & (z>=first[:,:,None]) & (z<=last[:,:,None])
    envelope = ndi.binary_fill_holes(envelope)
    before_regularization=int(envelope.sum())
    if minimum_feature>0:
        # Remove exterior fins too small for the manufacturing profile. This is
        # a general volume regularizer, with removed material explicitly counted.
        r=minimum_feature/2
        kernel_extent=int(np.ceil(r/pitch))+1
        offsets=np.stack(np.meshgrid(*([np.arange(-kernel_extent,kernel_extent+1)]*3),indexing="ij"),-1)*pitch
        ball=np.linalg.norm(offsets,axis=-1)<=r+pitch/2
        envelope=ndi.binary_opening(envelope,structure=ball)
        envelope=ndi.binary_fill_holes(envelope)
    if not envelope.any():
        raise GeometryInputError('EMPTY_REPAIRED_VOLUME','No material survives the required minimum feature size. Check input units/length; minimum-wall requirements were not relaxed.',
                                 input_dimensions_mm=mesh.extents,voxel_pitch_mm=pitch,minimum_feature_mm=minimum_feature,
                                 occupied_before_regularization=before_regularization,occupied_after_regularization=0)
    result = occupancy_mesh(envelope, origin, pitch)
    report = dict(method="normal/duplicate repair + voxel closing + flood fill + vertical envelope",
                  pitch_mm=pitch, spatial_uncertainty_mm=float(np.sqrt(3)*pitch),
                  raw_surface_voxels=int(surf.sum()), closed_voxels=int(closed.sum()),
                  filled_voxels=int(filled.sum()), retained_voxels=int(envelope.sum()),
                  discarded_disconnected_voxels=int(filled.sum()-main.sum()), voxel_components=n,
                  manufacturing_regularization_removed_voxels=before_regularization-int(envelope.sum()),minimum_feature_mm=minimum_feature,
                  conservative_outer_envelope=True, warning="Voxel reconstruction can remove fine details; deviations measured separately.")
    report['grid_budget']=budget
    return result,envelope,origin,report
