"""Connect isolated tray supports with measured rails inside body projection."""
import numpy as np
from scipy.spatial import cKDTree
import manifold3d as mf
from smartcar.geometry.solid import box,from_mesh,to_mesh,intersection_volume


def attachment_measurement(rail,support,profile):
    overlap=float(intersection_volume(rail,support))
    if overlap>profile.collision_volume_tolerance:
        return dict(volume_mm3=overlap,shared_face_area_mm2=None)
    # A support regularized to start exactly on the floor joins a bottom rail
    # through a face, with zero intersection volume. Measure that face from
    # the loss of boundary area on union; edge/point contact loses no area.
    joined=rail+support
    shared=max(0.,float((rail.surface_area()+support.surface_area()-joined.surface_area())/2))
    if len(joined.decompose())==1 and shared>=profile.minimum_wall**2:
        return dict(volume_mm3=overlap,shared_face_area_mm2=shared)
    return None


def connect_tray_supports(chassis,vehicle,floor,exclusions,profile):
    pieces=sorted(chassis.decompose(),key=lambda s:s.volume(),reverse=True)
    if len(pieces)<=1:return chassis,[]
    projected=from_mesh(vehicle).project()
    outline=mf.CrossSection(projected.to_polygons(),mf.FillRule.Positive).offset(-profile.sliding_clearance)
    # The service seam starts at floor + sliding_clearance. A bridge extending
    # above the tray floor consumes that clearance even when it has no collision.
    permitted=outline.extrude(profile.bottom_thickness).translate((0,0,floor-profile.bottom_thickness))-exclusions
    main=pieces[0];records=[]
    for island in pieces[1:]:
        target_mesh=to_mesh(main);start_mesh=to_mesh(island)
        low=target_mesh.vertices[:,2]<=floor+profile.numerical_tolerance
        target=target_mesh.vertices[low,:2]
        low=start_mesh.vertices[:,2]<=start_mesh.bounds[0,2]+profile.numerical_tolerance
        start=start_mesh.vertices[low,:2]
        if not len(target) or not len(start):raise ValueError('SUPPORT_CONNECTION_NO_BASE_SURFACE')
        # Use geometric edge samples, not just the AABB centers of the islands.
        distance,index=cKDTree(target).query(start)
        chosen=None
        for j in np.argsort(distance)[:64]:
            a=start[j];b=target[index[j]];length=float(np.linalg.norm(b-a))
            if length<profile.numerical_tolerance:continue
            width=profile.nominal_wall*2
            rail=box([[-width/2,-width/2,floor-profile.bottom_thickness],[length+width/2,width/2,floor]])
            angle=float(np.degrees(np.arctan2(*(b-a)[::-1])))
            rail=rail.rotate((0,0,angle)).translate((*a,0))^permitted
            main_attachment=attachment_measurement(rail,main,profile)
            island_attachment=attachment_measurement(rail,island,profile)
            if main_attachment is None or island_attachment is None:continue
            result=main+island+rail
            if len(result.decompose())!=1:continue
            chosen=result
            records.append(dict(start_xy=a.tolist(),target_xy=b.tolist(),length_mm=length,width_mm=width,
                                bounds_mm=to_mesh(rail).bounds.tolist(),added_mm3=float((rail-chassis).volume()),
                                main_attachment=main_attachment,island_attachment=island_attachment,
                                method='bottom rail clipped to current body projection and actual wheel/access cuts',connected_components_after=1))
            break
        if chosen is None:raise ValueError('SUPPORT_CONNECTION_NO_COLLISION_FREE_RAIL')
        main=chosen
    return main,records
