from pathlib import Path
import copy
import trimesh
from smartcar.geometry.mesh import mesh_stats


def validate_stl_exports(report,output,names,prefix="exported_STL:"):
    result=copy.deepcopy(report)
    for name in names:
        mesh=trimesh.load(Path(output)/(name+".stl"),force="mesh",process=True)
        stats=mesh_stats(mesh)
        passed=stats["watertight"] and stats["manifold"] and stats["winding_consistent"] and stats["component_count"]==1 and stats['degenerate_triangles']==0 and (stats['volume_mm3'] or 0)>0
        result["checks"].append(dict(check=prefix+name,status="PASS" if passed else "FAIL",measurements=stats))
    result["counts"]={s:sum(c["status"]==s for c in result["checks"]) for s in ["PASS","FAIL","WARNING"]}
    result["blocking_checks"]=[c["check"] for c in result["checks"] if c["status"]=="FAIL"]
    result["status"]="FAIL" if result["blocking_checks"] else "WARNING"
    return result
