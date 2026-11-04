"""Acceptance equivalence and one public default for open lower-face fitting."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from head_rgbd.motion_acceptance import batch_backtrack
from head_rgbd import cli
from head_rgbd.pipeline import reconstruct_locked, actual_configuration
from head_rgbd import motion_v2


def test_batched_gate_matches_independent_fixed_ray_decisions():
    calls=[]

    def forward(alpha):
        calls.append(alpha)
        return [alpha]*4

    def validate(i,alpha):
        # Lower error alone must never excuse losing an original ray.
        residual=np.array([1.,1.])+alpha*np.array([-1.-i,-1.-i])
        score=float((residual**2).mean())
        coverage=(i!=2 or alpha<=.125) and (i!=3 or alpha==0.)
        return dict(accepted=score<=1. and coverage,score=score,coverage=coverage)

    actual,count=batch_backtrack(range(4),forward,validate)
    reference={}
    for i in range(4):
        for alpha in (1.,.5,.25,.125,0.):
            info=validate(i,alpha)
            if info['accepted']:
                reference[i]=dict(info,scale=alpha)
                break
    assert actual==reference
    assert count==len(calls)==5
    assert actual[2]['scale']==.125
    assert actual[3]['scale']==0.


def test_no_eligible_frames_does_not_evaluate_mesh():
    selected,calls=batch_backtrack([],lambda _:pytest.fail('unexpected mesh evaluation'),lambda *_:None)
    assert selected=={} and calls==0


def test_original_ray_coverage_and_penalty_survive_shared_residual_query():
    points=np.array([[0.,0.,-.5],[.01,0.,-.5],[.02,0.,-.5]])
    corr=dict(ray=np.array([0,2]),hit=points[[0,2]]+np.array([0.,0.,.001]),target=points[[0,2]])
    with patch.object(motion_v2,'ray_surface',return_value=(corr,{})) as query:
        residual,covered=motion_v2.fixed_ray_residual(None,None,points)
    assert query.call_count==1
    np.testing.assert_array_equal(covered,[True,False,True])
    np.testing.assert_allclose(residual,[1.,8.,1.])
    with patch.object(motion_v2,'ray_surface',return_value=(corr,{})):
        score,_=motion_v2.fixed_ray_score(None,None,points)
        required_score,_=motion_v2.fixed_ray_score(None,None,points,covered)
    assert score==pytest.approx(22.)
    assert required_score==pytest.approx(1.)


def test_unaccepted_zero_scale_is_reported_without_claiming_success():
    selected,calls=batch_backtrack([1],lambda a:{1:a},lambda *_:dict(accepted=False))
    assert selected[1]==dict(accepted=False,scale=0.)
    assert calls==5


@pytest.mark.parametrize('weight', ['-1','nan','inf','-inf'])
def test_invalid_weight_is_rejected_before_capture(weight):
    with patch.object(cli,'command_run',side_effect=AssertionError('hardware accessed')) as command:
        with pytest.raises(SystemExit) as exc:
            cli.main(['run',f'--motion-lowerface-weight={weight}'])
    assert exc.value.code==2
    command.assert_not_called()


def test_public_defaults_and_full_chain_override_share_one_weight():
    run=cli.build_parser().parse_args(['run'])
    reconstruct=cli.build_parser().parse_args(['reconstruct','--sequence','sequence.json'])
    assert run.motion_lowerface_weight==reconstruct.motion_lowerface_weight==.5
    args=cli.build_parser().parse_args(['reconstruct','--sequence','sequence.json',
                                     '--chain','full','--motion-lowerface-weight','0'])
    calls=[]

    class Recorder:
        runtime=Path('runtime')
        def run(self,name,module,arguments,**_):
            calls.append((module,arguments))
            return 0

    reconstruct_locked(Recorder(),sequence=Path('sequence.json'),destination=Path('fake'),args=args)
    motion=[a for m,a in calls if m=='head_rgbd.motion_v2']
    assert len(motion)==2
    assert all(a[a.index('--lowerface-weight')+1]=='0.0' for a in motion)


def test_export_uses_saved_motion_configuration_instead_of_new_default(tmp_path):
    import json
    fit=tmp_path/'dense'
    fit.mkdir()
    (fit/'report.json').write_text(json.dumps(dict(field_logs={},shape_fusion=dict(mode='joint'))))
    motion=tmp_path/'motion_only'
    motion.mkdir()
    report=motion/'report.json'
    report.write_text(json.dumps(dict(motion_correction=dict(steps=80))))
    legacy=actual_configuration(fit)
    assert legacy['motion_lowerface_weight']==0.
    assert legacy['motion_acceptance_strategy']=='per_frame_legacy'
    report.write_text(json.dumps(dict(motion_correction=dict(steps=80,lowerface_weight=.5,
        lowerface_frames=['open_front_4'],acceptance_strategy='batch_by_shared_backtracking_scale',acceptance_forward_calls=3))))
    current=actual_configuration(fit)
    assert current['motion_lowerface_weight']==.5
    assert current['motion_lowerface_frames']==['open_front_4']
    assert current['motion_acceptance_forward_calls']==3
