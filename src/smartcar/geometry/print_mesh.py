"""Stabilize numerical STL topology before all independent design validators."""
import numpy as np
import trimesh
from smartcar.geometry.solid import to_mesh,from_mesh,cancel_collapsed_faces
from smartcar.geometry.mesh import MIN_TRIANGLE_AREA_MM2


def finalize_print_parts(parts,profile):
    """Preserve manufactured surfaces through the actual printable export path.

    A small vertex-error budget does not bound the resulting face-normal error.
    Simplifying the reconstructed shell here can create a thin ledge even after
    its local junction treatment passed. Preview meshes are simplified separately.
    """
    final={};records={}
    for name,solid in parts.items():
        print(f'STL stabilize {name}',flush=True)
        encoded=cancel_collapsed_faces(to_mesh(solid))
        small=int(np.sum(encoded.area_faces<MIN_TRIANGLE_AREA_MM2))
        broken=not encoded.is_watertight or not encoded.is_winding_consistent
        if small or broken:
            # A float32-encoded tiny chamfer can have nonzero height but an
            # area below the SAME mesh-validator threshold. Remove it only at
            # the encoding uncertainty, not the much coarser design tolerance.
            precision=np.finfo(np.float32).eps*max(float(np.abs(encoded.bounds).max()),1.)
            epsilon=precision*4
            # Condition the original double-precision Boolean solid BEFORE
            # float32 welding can destroy the seams of a planar bottom plate.
            # MeshFix cannot reliably infer these coincident physical faces.
            candidate,repair=stabilize_print_solid(solid.simplify(epsilon),profile)
            added=max(0.,(candidate-solid).volume());removed=max(0.,(solid-candidate).volume())
            budget=max(profile.collision_volume_tolerance,float(encoded.area)*epsilon)
            remaining=int(np.sum(to_mesh(candidate).area_faces<MIN_TRIANGLE_AREA_MM2))
            record=dict(before_triangles_below_area_limit=small,after_triangles_below_area_limit=remaining,
                area_limit_mm2=MIN_TRIANGLE_AREA_MM2,simplification_epsilon_mm=epsilon,
                added_mm3=added,removed_mm3=removed,precision_volume_budget_mm3=budget,stabilization=repair)
            record['before_encoding_topology_invalid']=broken
            records[name]=dict(subresolution_cleanup=record)
            if added+removed<=budget and not remaining and repair.get('status')!='FAIL':
                final[name]=candidate
            else:
                final[name]=solid
                records[name].update(status='FAIL',reason='PRINT_MESH_SUBRESOLUTION_CLEANUP_REJECTED')
        else:
            final[name],records[name]=stabilize_print_solid(solid,profile)
    return final,records


def stabilize_print_solid(solid,profile):
    mesh=cancel_collapsed_faces(to_mesh(solid))
    if mesh.is_watertight and mesh.is_winding_consistent:
        return from_mesh(mesh),dict(method='float32 weld and signed face cancellation',meshfix_used=False)
    precision=np.finfo(np.float32).eps*max(float(np.abs(mesh.bounds).max()),1.)
    from smartcar.geometry.conform_edges import conform_open_edges
    fixed,local_repair=conform_open_edges(mesh,precision*4)
    meshfix_used=False
    if not fixed.is_watertight or not fixed.is_winding_consistent:
        import pymeshfix
        vertices,faces=pymeshfix.clean_from_arrays(np.asarray(mesh.vertices,dtype=np.float64),np.asarray(mesh.faces,dtype=np.int32),
                                                  joincomp=False,remove_smallest_components=False)
        fixed=trimesh.Trimesh(vertices,faces,process=True);meshfix_used=True
    if not len(fixed.faces) or not fixed.is_watertight or not fixed.is_winding_consistent:
        return solid,dict(status='FAIL',reason='PRINT_MESH_TOPOLOGY_REPAIR_FAILED',meshfix_used=meshfix_used,local_repair=local_repair)
    candidate=from_mesh(fixed)
    added=max(0.,(candidate-solid).volume());removed=max(0.,(solid-candidate).volume())
    # Repair is allowed only within the volume uncertainty of float32 boundary
    # coordinates. Physical features cannot be dropped merely to close a mesh.
    budget=max(profile.collision_volume_tolerance,float(mesh.area)*precision*4)
    if added+removed>budget:
        return solid,dict(status='FAIL',reason='PRINT_MESH_REPAIR_CHANGED_GEOMETRY',rejected_symmetric_difference_mm3=added+removed,
                          precision_volume_budget_mm3=budget,meshfix_used=meshfix_used,local_repair=local_repair,action='retain original geometry for independent failed-candidate review')
    return candidate,dict(method='numerical boundary edge subdivision before validation; no component deletion or component joining',meshfix_used=meshfix_used,local_repair=local_repair,
                           symmetric_difference_mm3=added+removed,precision_volume_budget_mm3=budget,added_mm3=added,removed_mm3=removed)
