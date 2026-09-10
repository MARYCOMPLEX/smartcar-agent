"""Remove narrow planar prongs at reconstructed closure/shell junctions."""
import numpy as np
import manifold3d as mf
from smartcar.geometry.solid import to_mesh,from_mesh,union,box
from smartcar.structure.closure_ribs import rib_solid


def closure_regions(closure,profile):
    regions=[]
    for record in closure.get('connection_ribs',[]):
        bounds=to_mesh(rib_solid(record)).bounds.copy()
        bounds[:,:2]+=np.array([[-1,-1],[1,1]])*profile.nominal_wall
        regions.append(bounds)
    return regions


def clean_closure_junctions(body,closure,profile):
    regions=closure_regions(closure,profile)
    if not regions:return body,dict(layers=0,removed_mm3=0.)
    pitch=profile.shell_regularization_pitch
    lo=min(b[0,2] for b in regions);hi=max(b[1,2] for b in regions)
    slabs=[];layers=0
    bounds=to_mesh(body).bounds.copy()
    bounds[:,:2]+=np.array([[-1,-1],[1,1]])*profile.nominal_wall
    edges=np.linspace(lo,hi,int(np.ceil((hi-lo)/pitch))+1)
    print(f'JUNCTION reconstructing {len(edges)-1} planar layers',flush=True)
    for z0,z1 in zip(edges[:-1],edges[1:]):
        z=(z0+z1)/2
        active=[b for b in regions if b[0,2]<=z<=b[1,2]]
        if not active:continue
        section=body.slice(z)
        if section.is_empty():continue
        # A continuous planar opening removes thin diagonal grid prongs that
        # can survive binary 3-D reconstruction next to a restored cylinder.
        # Preserve already printable walls using the unchanged minimum wall.
        # The planar operation removes material; rebuilding finite-height
        # layers can shift sloping boundaries, measured separately below.
        eps=profile.numerical_tolerance
        # Reserve the in-plane uncertainty of the final regular-grid surface
        # reconstruction while keeping the physical minimum wall unchanged.
        radius=(profile.minimum_wall+np.sqrt(2)*pitch)/2+eps
        opened=(section.offset(-radius).offset(radius)^section).offset(-eps)
        # Scanline intersections of a prismatic face may differ by nanometers
        # between layers. Quantize only the retained boundary below the existing
        # numerical tolerance so equivalent layers share identical vertices.
        grid=profile.numerical_tolerance/4
        polygons=[np.round(polygon/grid)*grid for polygon in opened.to_polygons()]
        opened=mf.CrossSection(polygons,mf.FillRule.Positive)
        if opened.is_empty():continue
        # Build the retained planar material directly. Subtracting hundreds of
        # near-tangent cutter slabs from a sloping triangle mesh produces tiny
        # Boolean fragments that are expensive to repair on STL export.
        # Shared layer planes and quantized XY vertices give consistent seams.
        a=max(bounds[0,2],z0-eps) if z0==edges[0] else z0
        b=min(bounds[1,2],z1+eps) if z1==edges[-1] else z1
        slabs.append(opened.extrude(b-a).translate((0,0,a)));layers+=1
    cut_bounds=bounds.copy();cut_bounds[0,2]=lo;cut_bounds[1,2]=hi
    result=union([body-box(cut_bounds)]+slabs).simplify(profile.numerical_tolerance/2)
    print(f'JUNCTION reconstructed {layers} layers',flush=True)
    # Quantize the material as one closed surface before the caller restores
    # native interfaces. This eliminates sub-float32 Boolean seams between
    # planar layers while preserving the open interior as empty occupancy.
    from smartcar.geometry.voxelize import voxelize_closed_solid
    from smartcar.geometry.repair import rolling_sphere_mesh
    print('JUNCTION final grid reconstruction',flush=True)
    maximum=getattr(profile,'maximum_shell_voxel_cells',profile.maximum_voxel_cells)
    cells,origin,voxel=voxelize_closed_solid(to_mesh(result),pitch,max_cells=maximum)
    # Varying sections also create necks in Z. A planar operation cannot remove
    # those; apply the same nominal-wall rolling sphere to the restored junction.
    from scipy import ndimage as ndi
    r=profile.nominal_wall/2
    axis=np.arange(-int(np.ceil(r/pitch)),int(np.ceil(r/pitch))+1)*pitch
    offsets=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1)
    kernel=np.linalg.norm(offsets,axis=-1)<r-1e-9
    cells=ndi.binary_closing(cells,structure=kernel)
    cells=ndi.binary_opening(cells,structure=kernel)
    voxel['junction_rolling_diameter_mm']=2*r
    # Reconstruct the union of interior rolling spheres as a distance field.
    # Blurring binary labels can join two nearby walls with a very thin web;
    # this surface construction preserves the separation of the spheres.
    result=from_mesh(rolling_sphere_mesh(cells,origin,pitch,profile.nominal_wall))
    voxel['surface_reconstruction']='distance to interior rolling-sphere centers; no binary-label blur'
    voxel['reconstructed_ball_radius_mm']=profile.nominal_wall/2-pitch/2
    return result,dict(method='planar minimum-wall reconstruction across measured closure heights; independent final geometry validation required',
        layer_pitch_mm=float(edges[1]-edges[0]),configured_maximum_layer_pitch_mm=pitch,
        rolling_diameter_mm=profile.minimum_wall+np.sqrt(2)*pitch+2*profile.numerical_tolerance,
        required_minimum_wall_mm=profile.minimum_wall,layers=layers,
        net_removed_mm3=float(body.volume()-result.volume()),removed_mm3=float((body-result).volume()),
        added_mm3=float((result-body).volume()),vertical_boundary_uncertainty_mm=pitch/2+profile.numerical_tolerance,
        final_grid_reconstruction=voxel,
        regions_mm=[b.tolist() for b in regions])
