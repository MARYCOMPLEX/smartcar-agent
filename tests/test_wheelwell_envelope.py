import numpy as np
from smartcar.geometry.wheelwell_envelope import reconstruct_wheel_regions


def test_only_measured_wheel_regions_can_change_and_crown_width_is_used():
    # Box body with two lateral circular wheel cutouts; no previous chassis or
    # target file coordinates are involved in this independent fixture.
    origin=np.array([-42.,-95.,-3.]);pitch=1.
    x,y,z=np.meshgrid(np.arange(85)+origin[0],np.arange(191)+origin[1],np.arange(66)+origin[2],indexing='ij')
    body=(abs(x)<35)&(abs(y)<90)&(z>=10)&(z<=55)
    regions=np.zeros_like(body)
    for yy in [-55.,51.]:
        circle=(y-yy)**2+(z-18)**2<=18**2
        body&=~(circle&(abs(x)>21));regions|=(y-yy)**2+(z-18)**2<=(18*1.12)**2
    evidence=dict(source_bounds_mm=[[-40,-90,0],[40,90,55]],axles=[dict(y_mm=yy,z_mm=18,radius_mm=18) for yy in [-55.,51.]])
    result,record=reconstruct_wheel_regions(body,origin,pitch,evidence)
    assert record['applied'] and record['added_mm3']>0
    assert not np.any((body!=result)&~regions)
    assert np.array_equal(result[:,:,45:],body[:,:,45:])
    assert not result[z<10].any()
    assert result[(abs(x)<30)&(abs(y+55)<3)&(z==20)].all()


def test_unresolved_appearance_is_not_convexified():
    mask=np.zeros((20,40,30),bool);mask[3:16,3:36,5:25]=True
    result,record=reconstruct_wheel_regions(mask,[0,0,0],1,dict(source_bounds_mm=[[0,0,0],[19,39,29]],axles=[]))
    assert np.array_equal(result,mask) and not record['applied']


def test_narrow_underbody_rail_does_not_lower_wheelwell_belly():
    origin=np.array([-42.,-95.,-3.]);pitch=1.
    x,y,z=np.meshgrid(np.arange(85)+origin[0],np.arange(191)+origin[1],np.arange(66)+origin[2],indexing='ij')
    mask=(abs(x)<35)&(abs(y)<90)&(z>=10)&(z<=55)
    mask|=(abs(x)<4)&(abs(y)<75)&(z>=3)&(z<10)
    evidence=dict(source_bounds_mm=[[-40,-90,0],[40,90,55]],axles=[dict(y_mm=yy,z_mm=18,radius_mm=18) for yy in [-55.,51.]])
    result,record=reconstruct_wheel_regions(mask,origin,pitch,evidence)
    assert all(zone['belly_z_mm']==10 for zone in record['zones'])
    assert not result[(abs(x)>5)&(z<10)].any()
