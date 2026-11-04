import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from head_rgbd.cli import build_parser
from head_rgbd.geometry_d import Settings
from head_rgbd.pipeline import load_defaults, reconstruct_locked, actual_configuration
from head_rgbd.surface_v2 import safe_loop_base, loop_support_exclusion


def test_public_commands_share_configuration_and_independent_budgets():
    parser = build_parser()
    run = parser.parse_args(['run'])
    rebuild = parser.parse_args(['reconstruct', '--sequence', 'sequence.json'])
    config = load_defaults()['reconstruction']
    for key in ('shape_fusion', 'fast', 'chain', 'postfit', 'mid_rounds', 'reuse_field',
                'information_measure', 'support_min_information', 'coverage_allowance_ratio',
                'loop_data_selection', 'loop_target_blend', 'front_steps', 'steps', 'bending_length_mm'):
        assert getattr(run, key) == getattr(rebuild, key) == config[key]
    calls = []
    class Recorder:
        runtime = Path('runtime')
        def run(self, name, module, arguments):
            calls.append((name, [str(x) for x in arguments]))
            return 0
    reconstruct_locked(Recorder(), sequence=Path('sequence.json'), destination=Path('out'), args=run)
    assert [x[0] for x in calls] == ['identity_front_anchor', 'shared_identity_joint',
                                   'fixed_identity_motion', 'guarded_measured_dense_surface']
    assert calls[0][1][calls[0][1].index('--steps')+1] == '240'
    assert calls[1][1][calls[1][1].index('--steps')+1] == '240'
    assert '--loop-data-selection' not in calls[-1][1]
    assert '--reuse-anchor' not in sum([x[1] for x in calls], [])


def test_loop_uses_target_immediately_when_safe():
    P = sparse.eye(3, format='csr')
    base = np.array([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
    faces = np.array([[0, 1, 2]])
    with patch('head_rgbd.hierarchy_d.safety', return_value=(True, {'checked': True})) as check:
        _, _, result = safe_loop_base(P, P, base, base, base, faces, np.zeros(3), Settings())
    assert check.call_count == 1
    assert result['blend'] == result['target_blend'] == .5
    assert result['fallback_reason'] is None


def test_loop_stencil_support_matches_barycentric_information_without_inventing_evidence():
    support = np.array([[True, False, True], [False, False, True]])
    original = support.copy()
    excluded = np.array([False, False, True])
    mask, minimum = loop_support_exclusion(excluded, support, Settings(support_min_information=.5))
    assert minimum == 1
    np.testing.assert_array_equal(mask, [False, True, True])
    np.testing.assert_array_equal(support, original)
    legacy, minimum = loop_support_exclusion(excluded, support, Settings())
    assert minimum == 2
    assert legacy.all()


def test_front_budget_override_does_not_reduce_joint_steps():
    args = build_parser().parse_args(['reconstruct', '--sequence', 'sequence.json', '--front-steps', '120'])
    calls = []
    class Recorder:
        runtime = Path('runtime')
        def run(self, name, module, arguments):
            calls.append([str(x) for x in arguments])
            return 0
    reconstruct_locked(Recorder(), sequence=Path('sequence.json'), destination=Path('out'), args=args)
    assert calls[0][calls[0].index('--steps')+1] == '120'
    assert calls[1][calls[1].index('--steps')+1] == '240'


def test_configuration_uses_actual_stage_results_even_on_reexport(tmp_path):
    fit = tmp_path / 'dense'
    fit.mkdir()
    for folder, steps in [('front_anchor', 123), ('guarded', 337)]:
        p = tmp_path / folder
        p.mkdir()
        (p/'report.json').write_text(json.dumps(dict(steps=steps, device='cpu', points_per_frame=777,
                                                    expression_components=30, postfit='minimal')))
    (fit/'report.json').write_text(json.dumps(dict(shape_fusion=dict(mode='joint'),
        fast_path=dict(enabled=True, mid_frequency_rounds=1), loop_base=dict(target_blend=.5, blend=.25,
        fallback_reason='normal_change', data_selection=None), field_logs=dict(
        front_anchor_regularized=dict(settings=dict(bending_length_mm=8),
                                     mid_frequency=dict(settings=dict(points=5000, bending_length_mm=6))),
        regularized=dict(reused=True)))))
    config = actual_configuration(fit)
    assert config['front_steps'] == 123 and config['steps'] == 337
    assert config['loop_actual_blend'] == .25 and config['loop_target_blend'] == .5
    assert config['bending_length_mm'] == 6 and config['low_bending_length_mm'] == 8
    assert config['shape_fusion'] == 'joint' and config['chain'] == 'lean'
    assert config['postfit'] == 'minimal' and config['device'] == 'cpu'


def test_loop_backtracks_only_on_geometric_failure():
    P = sparse.eye(3, format='csr')
    base = np.array([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
    faces = np.array([[0, 1, 2]])
    with patch('head_rgbd.hierarchy_d.safety', side_effect=[
            (False, {'reason': 'total_vector_displacement_bound'}), (True, {'checked': True})]):
        _, _, result = safe_loop_base(P, P, base, base, base, faces, np.zeros(3), Settings())
    assert result['blend'] == .25
    assert result['target_blend'] == .5
    assert result['fallback_reason'][0]['check']['reason'] == 'total_vector_displacement_bound'


def test_loop_rejects_an_unsafe_control_even_at_zero_blend():
    import pytest
    P = sparse.eye(3, format='csr')
    base = np.array([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
    with patch('head_rgbd.hierarchy_d.safety', return_value=(False, {'reason': 'physical_slope'})):
        with pytest.raises(ValueError, match='no feasible Loop'):
            safe_loop_base(P, P, base, base, base, np.array([[0, 1, 2]]), np.zeros(3), Settings())
