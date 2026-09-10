from pathlib import Path
from types import SimpleNamespace
import numpy as np
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,cylinder,intersection_volume
from smartcar.structure.battery_tray import battery_tray


def test_raised_tray_cannot_leave_thin_channel_roof():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    floor=2.4
    battery=SimpleNamespace(id='battery',bounds=np.array([[-14.,-25,5.05],[14.,25,21.35]]))
    tray,slots,record=battery_tray(battery,floor,p)
    chassis=(box([[-20,-30,0],[20,30,floor]])+tray)-slots
    # A real vertical probe beside the pack measures the whole passage through
    # the raised retaining wall, independent of the returned slot's own bounds.
    probe=cylinder(.1,-1,battery.bounds[0,2]+p.tray_wall_height+.1,(-14.2,0))
    assert intersection_volume(chassis,probe)<p.collision_volume_tolerance
