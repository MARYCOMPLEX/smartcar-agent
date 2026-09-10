import numpy as np
import trimesh
from smartcar.geometry.solid import intersection_volume,to_mesh
from smartcar.render import render_meshes


def collision_visualization(run,parts,instances,profile):
    intersections=[]
    for inst in instances:
        for name,solid in parts.items():
            overlap=inst.proxy()^solid
            if overlap.volume()>profile.collision_volume_tolerance:
                m=to_mesh(overlap);m.visual.face_colors=[240,40,55,255]
                intersections.append(("collision_"+inst.id+"_"+name,m))
    scene=trimesh.Scene()
    for name,m in intersections:scene.add_geometry(m.copy().apply_scale(.001),node_name=name)
    if intersections:scene.export(run/"12_validation/collisions.glb")
    render_meshes(run/"12_validation/collisions.png",[("body",to_mesh(parts["body"]))]+intersections,
                  f"Measured collision regions ({len(intersections)} pairs)",transparent=True)
    return len(intersections)
