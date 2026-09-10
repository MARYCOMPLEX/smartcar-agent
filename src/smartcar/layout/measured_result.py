"""Attach independent measurements to solver results without changing poses."""
def attach_validation(instances,report):
    for inst in instances:
        related=[];gaps=[]
        for check in report['checks']:
            label=check['check'];tokens=label.split(':')
            if inst.id not in tokens:continue
            related.append(dict(check=label,status=check['status']))
            if label.startswith('hardware_clearance:') and check['status']=='PASS':gaps.append(check['minimum_gap_mm'])
            if label.startswith('hardware_structure:') and not check.get('contact_allowed',False) and 'gap_mm' in check:
                gaps.append(check['gap_mm'])
            if label.startswith('wheel_rotation:'):gaps.append(check['minimum_swept_gap_mm'])
        inst.metadata['candidate_envelope_slack_mm']=inst.minimum_clearance_mm
        inst.minimum_clearance_mm=min(gaps) if gaps else None
        inst.metadata['minimum_clearance_definition']='measured conservative lower bound against non-mating hardware and print surfaces; intentional mounting contact and fixed wheel socket mating excluded'
        inst.constraints['independent_validation']=related
