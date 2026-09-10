"""Reconstruct printable shell material, then restore exact screw interfaces."""
import numpy as np
from scipy import ndimage as ndi
from smartcar.geometry.repair import occupancy_mesh
from smartcar.geometry.solid import from_mesh,to_mesh,cylinder,union,box
from smartcar.geometry.voxelize import voxelize_closed_solid


def regularize_shell(body,closure,profile,restoration_envelope=None):
    body=body.set_tolerance(profile.numerical_tolerance/2)
    pitch=profile.shell_regularization_pitch
    mesh=to_mesh(body)
    # Binary shell reconstruction does not allocate the layout distance field
    # and erosion buffers. Give this stage its own explicit resource budget,
    # without changing either its physical pitch or any acceptance threshold.
    maximum_cells=getattr(profile,'maximum_shell_voxel_cells',getattr(profile,'maximum_voxel_cells',80000000))
    occupied,origin,voxel_report=voxelize_closed_solid(mesh,pitch,max_cells=maximum_cells)
    voxel_report['maximum_cells']=maximum_cells
    # Use nominal wall as the rolling-sphere diameter. The independent minimum
    # wall threshold stays unchanged; discretization is recorded, not hidden.
    radius=profile.nominal_wall/2
    n=int(np.ceil(radius/pitch))
    axes=[np.arange(-n,n+1)*pitch]*3
    offsets=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    # Occupancy is sampled at cell centers. Requiring both endpoints of a
    # diameter exactly equal to the nominal wall asks for an extra whole cell
    # and erases a valid flat wall. Use the sphere's strict interior support.
    kernel=np.linalg.norm(offsets,axis=-1)<radius-1e-9
    opened=ndi.binary_opening(occupied,structure=kernel)
    candidate=from_mesh(occupancy_mesh(opened,origin,pitch)).set_tolerance(profile.numerical_tolerance/2)
    # Avoid intersecting almost coincident reconstructed surfaces. Critical
    # clearance reserves include the reconstruction uncertainty, and every final
    # hardware/path/clearance test is rerun against this actual reconstructed body.
    floor=closure['seam_z'];end=floor+profile.sliding_clearance+profile.screw_fit['thread_length']
    keep=union([cylinder(profile.fastener_boss_radius,floor+profile.sliding_clearance,end,xy) for xy in closure['centers_xy']])
    from smartcar.structure.closure_ribs import rib_solid
    ribs=union([rib_solid(record) for record in closure.get('connection_ribs',[])])
    # Restore measured structural ribs as well as the native fastener interface;
    # restoring bosses alone can undo connectivity after the rolling operation.
    restored_ribs=ribs^body
    if restoration_envelope is not None:
        # Native ribs must not reintroduce the outer thin fins that the rolling
        # operation removed. Restore their interior portion, overlapping the
        # manufactured wall from within the measured safe exterior envelope.
        restored_ribs=restored_ribs^restoration_envelope
    keep=keep+restored_ribs
    candidate=candidate+keep
    from smartcar.structure.junction_cleanup import clean_closure_junctions
    candidate,junction_report=clean_closure_junctions(candidate,closure,profile)
    # Retain the exact screw interface after subtractive junction cleanup.
    # Do not restore tall ribs again: that would restore the same narrow prongs.
    candidate=candidate+union([cylinder(profile.fastener_boss_radius,floor+profile.sliding_clearance,end,xy) for xy in closure['centers_xy']])
    # Drill through the restored boss. Ending at the old unshifted top leaves a
    # roof only as thick as the mating clearance, which is not a printable wall.
    from smartcar.structure.closure_bores import closure_bores
    holes,bore_records=closure_bores(candidate,closure,profile)
    candidate=candidate-holes
    bounds=to_mesh(candidate).bounds.copy();bounds[0,2]=floor+profile.sliding_clearance
    bounds[:,:2]+=np.array([[-1,-1],[1,1]])*pitch;bounds[1,2]+=pitch
    # Restore the exact mating seam; morphology must not consume lid clearance.
    candidate=candidate^box(bounds)
    return candidate,dict(method='scanline occupancy and rolling-sphere reconstruction; native screw bosses/bores and seam restored before independent validation',
                          voxel_pitch_mm=pitch,rolling_diameter_mm=radius*2,net_volume_reduction_mm3=body.volume()-candidate.volume(),
                          before_mm3=body.volume(),after_mm3=candidate.volume(),boundary_uncertainty_mm=float(np.sqrt(3)*pitch),voxelization=voxel_report,
                          closure_junction_cleanup=junction_report,closure_bores=bore_records)
