from pathlib import Path
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,intersection_volume
from smartcar.structure.cavity import remove_thin_internal_webs


def test_thin_web_removed_only_in_allowed_interior():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/"config/manufacturing.json")
    a=[[0,0,0],[10,10,10]];b=[[0,11,0],[10,20,10]]
    allowed=box([[2,2,2],[8,18,8]])
    cut,records=remove_thin_internal_webs([a,b],allowed,p)
    assert records
    assert intersection_volume(cut,box([[2,10,2],[8,11,8]]))>35.9
    assert intersection_volume(cut,box([[0,0,0],[1,20,10]]))==0
