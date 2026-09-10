from __future__ import annotations
import numpy as np
import trimesh
from scipy.spatial import cKDTree

MIN_TRIANGLE_AREA_MM2=1e-10


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=True)
    if not len(mesh.faces): raise ValueError("EMPTY_APPEARANCE")
    return mesh


def mesh_stats(mesh, include_components=True):
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    area = mesh.area_faces
    result = dict(vertices=len(mesh.vertices), triangles=len(mesh.faces), bounds_mm=mesh.bounds,
                  dimensions_mm=mesh.extents, watertight=mesh.is_watertight,
                  winding_consistent=mesh.is_winding_consistent,
                  manifold=bool(np.all(edge_counts == 2)), boundary_edges=int(sum(edge_counts == 1)),
                  nonmanifold_edges=int(sum(edge_counts > 2)), surface_area_mm2=float(mesh.area),
                  volume_mm3=float(mesh.volume) if mesh.is_volume else None,
                  degenerate_triangles=int(sum(area < MIN_TRIANGLE_AREA_MM2)), euler_number=mesh.euler_number)
    if include_components:
        components = sorted(mesh.split(only_watertight=False), key=lambda m: m.area, reverse=True)
        result["component_count"] = len(components)
        result["components"] = [dict(index=i, faces=len(m.faces), area_mm2=m.area, bounds_mm=m.bounds,
                                      watertight=m.is_watertight, volume_mm3=float(m.volume) if m.is_volume else None) for i,m in enumerate(components)]
    return result


def sample_surface(mesh, count=5000):
    return trimesh.sample.sample_surface(mesh, count, seed=42)[0]


def symmetry_error(mesh, axis):
    p = sample_surface(mesh, 10000)
    mirrored = p.copy()
    mirrored[:,axis] = 2 * mesh.bounds[:,axis].mean() - mirrored[:,axis]
    d = cKDTree(p).query(mirrored)[0]
    return dict(mean_mm=float(d.mean()), p95_mm=float(np.quantile(d, .95)))


def transformed(mesh, matrix):
    return mesh.copy().apply_transform(matrix)
