"""Verify strap channels against the actual final print solids."""
from smartcar.geometry.solid import box,intersection_volume,union


def validate_battery_passages(parts,records,profile):
    material=union(list(parts.values()));checks=[]
    for rec in records:
        bounds=rec.get('strap_passage_bounds',[])
        if len(bounds)!=2:
            checks.append(dict(check='battery_strap_access:'+rec['hardware'],status='FAIL',reason='Two physical strap channels are required.'))
        for index,bound in enumerate(bounds):
            obstruction=intersection_volume(material,box(bound))
            checks.append(dict(check=f"battery_strap_access:{rec['hardware']}:{index}",
                               status='PASS' if obstruction<=profile.collision_volume_tolerance else 'FAIL',
                               intersection_mm3=obstruction,passage_bounds=bound,
                               sliding_clearance_mm=rec['strap_sliding_clearance_mm'],
                               method='reserved strap passage vs every final generated print part'))
    return checks
