from __future__ import annotations
import itertools
import numpy as np
from ortools.sat.python import cp_model
from smartcar.geometry.collision import aabb_distance,overlaps,expand
from smartcar.layout.scoring import total_score
from smartcar.layout.reservations import structural_bounds


def solve_layout(groups,profile):
    if any(not v for v in groups.values()):
        return None,dict(status="NO_CANDIDATES",empty=[k for k,v in groups.items() if not v])
    model=cp_model.CpModel(); vars={}
    for name,cs in groups.items():
        vars[name]=[model.new_bool_var(f"{name}_{i}") for i in range(len(cs))]
        model.add_exactly_one(vars[name])
    conflicts=0;compatible=0
    paircost=[]
    for a,b in itertools.combinations(groups,2):
        for i,x in enumerate(groups[a]):
            for j,y in enumerate(groups[b]):
                # Same XY footprints cannot be assembled directly down onto a
                # common tray, even if different Z values hide final collisions.
                bx=structural_bounds(x,profile);by=structural_bounds(y,profile);bx[0,2]=by[0,2]=-1e6;bx[1,2]=by[1,2]=1e6
                collision=overlaps(expand(bx,profile.rigid_clearance),by)
                # Lead length is a necessary straight-line bound; true connector
                # routing remains unverified without connector semantic frames.
                distance=np.linalg.norm(x.bounds.mean(0)-y.bounds.mean(0))
                lead=x.definition.raw.get("service_constraints",{}).get("lead_free_length_mm",{}).get("value",1e6)
                if collision or distance>lead:
                    model.add_bool_or([vars[a][i].Not(),vars[b][j].Not()]);conflicts+=1
                else:compatible+=1
    # Wiring distance to drive centers can be added as a unary objective by the
    # caller. Record all constituent terms in each instance.
    model.minimize(sum(round(total_score(c.score)*10000)*vars[n][i] for n,cs in groups.items() for i,c in enumerate(cs)))
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=profile.solver_time_seconds
    solver.parameters.num_search_workers=1;solver.parameters.random_seed=profile.seed
    status=solver.solve(model)
    info=dict(status=solver.status_name(status),candidate_counts={k:len(v) for k,v in groups.items()},incompatibility_constraints=conflicts,compatible_pairs=compatible,
              objective=solver.objective_value if status in [cp_model.OPTIMAL,cp_model.FEASIBLE] else None,wall_time_seconds=solver.wall_time)
    if status not in [cp_model.OPTIMAL,cp_model.FEASIBLE]:return None,info
    result=[cs[next(i for i,v in enumerate(vars[n]) if solver.boolean_value(v))] for n,cs in groups.items()]
    return result,info
