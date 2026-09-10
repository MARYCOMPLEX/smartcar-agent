import numpy as np


def hardware_score(bounds,vehicle_bounds,role):
    center=np.asarray(bounds).mean(0); span=np.ptp(vehicle_bounds,axis=0)
    # Geometric height only: no fabricated masses or claimed real center of mass.
    return dict(geometric_height=float(center[2]/span[2]),lateral_imbalance=float(abs(center[0])/span[0]),
                end_penalty=float(abs(center[1])/span[1]),cavity_volume=float(np.prod(np.ptp(bounds,axis=0))/np.prod(span)))


def total_score(parts):
    weights=dict(geometric_height=3.,lateral_imbalance=2.,end_penalty=.3,cavity_volume=.2,
                 wiring=2.,appearance_damage=5.,wheelbase_reward=-2.,track_exposure=.5)
    return sum(weights.get(k,1.)*v for k,v in parts.items())
