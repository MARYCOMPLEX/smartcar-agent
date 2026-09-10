from pathlib import Path
from types import SimpleNamespace
import numpy as np
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.domain.assembly import placed
from smartcar.geometry.sdf import DesignableVolume
from smartcar.geometry.wheel import arch_radius
from smartcar.layout.hardware_candidates import generate_candidates


def test_pack_above_tire_reserves_printable_tray_floor():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    battery=SimpleNamespace(role='battery',bounding_box=np.array([[-7.,-15,-5],[7.,15,5]]),features={})
    wheel=SimpleNamespace(role='passive_wheel',bounding_box=np.array([[-6.5,-18.5,-18.5],[6.5,18.5,18.5]]),
                          raw={'rigid_body':{'nominal_diameter_mm':37,'nominal_width_mm':13}},features={})
    obstacle=placed(wheel,np.eye(3),[0,0,18.5],'wheel','parallel')
    mask=np.zeros((49,79,151),bool);mask[3:-3,3:-3,3:-3]=True
    volume=DesignableVolume(mask,[-24*.6,-39*.6,-3*.6],.6,p.nominal_wall)
    vehicle=SimpleNamespace(bounds=np.array([[-13.,-22,0],[13.,22,88]]),extents=np.array([26.,44,88]))
    candidates,_=generate_candidates(battery,volume,vehicle,[obstacle],2.4,p,'spatial_free')
    assert candidates
    for inst in candidates:
        overlap=np.all((inst.bounds[0,:2]<obstacle.bounds[1,:2])&(inst.bounds[1,:2]>obstacle.bounds[0,:2]))
        if overlap:
            assert inst.bounds[0,2]>=18.5+arch_radius(18.5,p)+p.nominal_wall-1e-8
