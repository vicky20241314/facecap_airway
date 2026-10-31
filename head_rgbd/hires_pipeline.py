"""Explicit research pipeline; C guarded remains the application default."""
import json
import shutil
from pathlib import Path
from head_rgbd.paths import resolve_runtime_dir, verify_assets, asset_paths
from head_rgbd.reporting import StageRunner, require_stage_success


def add_parser(sub, defaults):
    p=sub.add_parser('depth-hires',help='research-only depth regularization and actual dense mesh reconstruction')
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--capture',type=Path)
    inputs.add_argument('--sequence',type=Path)
    p.add_argument('--initial-fit',type=Path,help='optional completed C initialization bound to --sequence')
    p.add_argument('--joint-initial-fit',type=Path,help='optional existing all-closed-view initialization, exact same sequence')
    p.add_argument('--baseline',type=Path,required=True,help='read-only copied v0.1 capture-run root for comparison')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--research-only',action='store_true',required=True)
    p.add_argument('--include-joint',action='store_true',help='also fit all-closed-view coarse identity and joint detail candidate')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--detail-points',type=int,default=5000)
    p.add_argument('--solver',choices=('least-squares','qp'),default='qp')
    p.add_argument('--dense-mode',choices=('absolute','hierarchical'),default='absolute')
    p.add_argument('--dense-residual-bound-mm',type=float,default=.5)
    p.add_argument('--regularized-fit',type=Path,help='optional verified previous D guarded control field')
    p.add_argument('--joint-regularized-fit',type=Path,help='optional verified previous D joint control field')
    p.add_argument('--gradient-length-mm',type=float,default=3.)
    p.add_argument('--bending-length-mm',type=float,default=4.)
    p.set_defaults(func=run)


def run(args):
    if args.initial_fit and not args.sequence:
        raise ValueError('--initial-fit requires the exact --sequence; use --capture without it for cold start')
    runtime=resolve_runtime_dir(args.runtime)
    assets=verify_assets(asset_paths(runtime))
    if any(a['status']!='ok' for a in assets):raise ValueError('runtime assets failed SHA256 verification')
    runner=StageRunner(args.output,runtime=runtime,purpose='saved_real_capture_depth_hires_research')
    from head_rgbd.audit import sha
    snapshot=runner.output/'source_snapshot'/'head_rgbd'
    snapshot.mkdir(parents=True)
    sources=sorted(Path(__file__).parent.glob('*.py'))
    for source in sources:shutil.copyfile(source,snapshot/source.name)
    runner.report['source_snapshot']=str(snapshot)
    runner.report['source_sha256']={p.name:sha(p) for p in sources}
    runner.flush()
    try:
        if args.initial_fit:
            sequence=args.sequence.resolve();initial=args.initial_fit.resolve()
        else:
            initial_root=runner.output/'initialization'
            source=['--capture',args.capture] if args.capture else ['--sequence',args.sequence]
            require_stage_success(runner,runner.run('initialize_identity_motion_expression','head_rgbd.cli',[
                '--runtime',runtime,'run',*source,'--output',initial_root,'--device',args.device,'--steps',str(args.steps)]))
            sequence=initial_root/'prepared'/'sequence.json' if args.capture else args.sequence.resolve()
            initial=initial_root/'reconstruction'/'guarded'
        common=['--sequence',sequence,'--runtime',runtime,'--research-only','--points',str(args.detail_points),'--solver',args.solver,
                '--dense-mode',args.dense_mode,'--dense-residual-bound-mm',str(args.dense_residual_bound_mm),
                '--gradient-length-mm',str(args.gradient_length_mm),'--bending-length-mm',str(args.bending_length_mm)]
        guarded=runner.output/'guarded'
        require_stage_success(runner,runner.run('regularized_and_dense_guarded','head_rgbd.fit_d',[
            *common,'--initial-fit',initial,'--mode','guarded','--output',guarded,
            *(['--regularized-fit',args.regularized_fit.resolve()] if args.regularized_fit else [])]))
        joint=None
        if args.include_joint:
            joint_initial=args.joint_initial_fit.resolve() if args.joint_initial_fit else runner.output/'joint_initialization'
            if not args.joint_initial_fit:
                require_stage_success(runner,runner.run('all_closed_view_coarse_identity_research','head_rgbd.fit_c',[
                    '--sequence',sequence,'--runtime',runtime,'--output',joint_initial,'--device',args.device,
                    '--steps',str(args.steps),'--points','1200','--expression-components','30']))
            joint=runner.output/'joint'
            require_stage_success(runner,runner.run('regularized_and_dense_joint_research','head_rgbd.fit_d',[
                *common,'--initial-fit',joint_initial,'--mode','joint-research','--output',joint,
                *(['--regularized-fit',args.joint_regularized_fit.resolve()] if args.joint_regularized_fit else [])]))
        comparison=runner.output/'comparison'
        compare_args=['--sequence',sequence,'--baseline',args.baseline.resolve(),'--guarded',guarded,'--output',comparison]
        if joint:compare_args+=['--joint',joint]
        require_stage_success(runner,runner.run('same_capture_geometry_comparison','head_rgbd.evaluate_d',compare_args))
        report=json.loads((comparison/'comparison.json').read_text(encoding='utf-8'))
        code=runner.finish(statuses=dict(geometry_effect_acceptance=report['acceptance']['D_guarded_dense']['overall'],
                                        candidate_promotion='not_promoted',
                                        independent_accuracy='unverified',saved_real_capture_pipeline='passed'),
            summary=dict(sequence=str(sequence),guarded=str(guarded),joint=str(joint) if joint else None,
                         comparison=str(comparison/'comparison.json'),acceptance=report['acceptance'],
                         real_camera_capture_this_run=False,asset_verification=assets))
    except BaseException as exc:
        runner.report['error']=f'{type(exc).__name__}: {exc}'
        runner.finish()
        raise
    print(json.dumps({'run_report':str(runner.report_path)}),flush=True)
    return code
