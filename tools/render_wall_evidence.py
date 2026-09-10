"""Read-only orthogonal sections around a measured minimum-wall witness."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from smartcar.geometry.rays import all_ray_hits
from smartcar.geometry.solid import from_mesh
from smartcar.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--part', default='body')
    args = parser.parse_args()
    root = args.run
    report_path = root / 'output/validation_report.json'
    mesh_path = root / f'output/{args.part}.stl'
    if not report_path.exists():
        report_path = root / '12_validation/validation_report.json'
        mesh_path = root / f'09_structure/{args.part}.stl'
    check = next(c for c in read_json(report_path)['checks']
                 if c['check'] == f'wall_thickness:{args.part}')
    witness = np.asarray(check['measurements']['examples'][0], dtype=float)
    mesh = trimesh.load(mesh_path, force='mesh')
    # Diagnostic ray only: repeat the original sampling offset, obtained from
    # the frozen profile instead of assuming a new manufacturing tolerance.
    profile_path = root / '01_input/provenance.json'
    if not profile_path.exists():
        profile_path = root.parent.parent / '01_input/provenance.json'
    profile = read_json(profile_path)['manufacturing']
    # Reconstruct the original sample/face association. Nearest-surface
    # reprojection at a tiny junction can select an adjacent face with a
    # different normal, even when the displacement is below a micrometer.
    random_count = check['measurements'].get('random_surface_samples', 1200)
    sampled, sampled_faces = trimesh.sample.sample_surface(mesh, random_count, seed=profile['seed'])
    distance = np.linalg.norm(sampled - witness, axis=1)
    k = int(np.argmin(distance)) if len(distance) else None
    if k is not None and distance[k] <= 1e-8:
        point = sampled[k]
        face = int(sampled_faces[k])
        offset = float(distance[k])
        association = 'original seeded sample and its own face'
    else:
        distance = np.linalg.norm(mesh.triangles_center - witness, axis=1)
        face = int(np.argmin(distance))
        if distance[face] <= 1e-8:
            point = mesh.triangles_center[face]
            offset = float(distance[face])
            association = 'original critical-region triangle centroid and its own face'
        else:
            points, offsets, faces = trimesh.proximity.closest_point(mesh, [witness])
            point, offset, face = points[0], float(offsets[0]), int(faces[0])
            association = 'nearest-surface reprojection; original face could not be reconstructed'
    normal = mesh.face_normals[face]
    epsilon = profile['numerical_tolerance']
    hits, _, _ = all_ray_hits(mesh, [point - normal * epsilon], [-normal])
    distances = np.linalg.norm(hits - (point - normal * epsilon), axis=1) + epsilon
    eligible = np.flatnonzero(distances > epsilon * 2)
    if not len(eligible):
        raise ValueError('Measured witness ray has no resolved exit')
    selected = eligible[np.argmin(distances[eligible])]
    hit = hits[selected]
    measured = float(distances[selected])
    half_span = profile['nominal_wall'] * 3
    solid = from_mesh(mesh)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1))
    # Rigid rotations put each requested section normal on the slicing Z axis.
    projections = [('XY', (0, 1, 2), (0, 0, 0), np.eye(3)),
                   ('XZ', (0, 2, 1), (90, 0, 0),
                    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
                   ('YZ', (1, 2, 0), (0, -90, 0),
                    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))]
    labels = ['X', 'Y', 'Z']
    for ax, (name, (u, v, fixed), rotation, matrix) in zip(axes, projections):
        rotated_point = matrix @ point
        section = solid.rotate(rotation).slice(float(rotated_point[2]))
        for polygon in section.to_polygons():
            lifted = np.column_stack([polygon, np.full(len(polygon), rotated_point[2])]) @ matrix
            closed = np.vstack([lifted, lifted[:1]])
            ax.plot(closed[:, u], closed[:, v], color='#386f87', linewidth=1)
        ax.plot([point[u], hit[u]], [point[v], hit[v]], color='#c53b35', linewidth=2)
        ax.scatter(point[u], point[v], s=25, c='#c53b35', label='Sampled entry')
        ax.scatter(hit[u], hit[v], s=28, marker='x', c='#542b22', label='Ray exit (projected)')
        ax.set_xlim(point[u] - half_span, point[u] + half_span)
        ax.set_ylim(point[v] - half_span, point[v] + half_span)
        ax.set_aspect('equal')
        ax.grid(alpha=.22)
        ax.set_xlabel(f'{labels[u]} / mm')
        ax.set_ylabel(f'{labels[v]} / mm')
        ax.set_title(f'{name} section: {labels[fixed]} = {point[fixed]:.4f} mm')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle(f'{args.part}: witness ray {measured:.6f} mm; original minimum '
                 f'{check["measurements"]["minimum_mm"]:.6f} mm | '
                 f'required {check["required_mm"]:.3f} mm | {check["status"]}', fontsize=12)
    fig.text(.5, .015, 'Actual exported geometry. The 3-D exit is projected onto each section; '
             'it need not lie on that section. Original mesh and validation report are unchanged.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    write_json(args.output.with_suffix('.json'), dict(
        kind='read-only measured wall sections', mesh=str(mesh_path), report=str(report_path),
        profile=str(profile_path), witness_surface_distance_mm=offset,
        face_association=association, face_index=face,
        entry_mm=point.tolist(), exit_mm=hit.tolist(), normal=normal.tolist(),
        measured_ray_mm=measured, required_mm=check['required_mm'],
        original_minimum_mm=check['measurements']['minimum_mm'],
        original_check_status=check['status']))
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
