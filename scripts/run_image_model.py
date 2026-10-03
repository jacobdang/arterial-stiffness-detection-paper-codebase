#!/usr/bin/env python3
"""Launch study image training or validation-ranked checkpoint evaluation."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    train=sub.add_parser('train',help='Train a bilateral or single-eye image model')
    train.add_argument('--image-root',type=Path,required=True)
    train.add_argument('--split-root',type=Path,required=True)
    train.add_argument('--output-dir',type=Path,required=True)
    train.add_argument('--model',default='image_only_tinynet_c')
    train.add_argument('--occlusion-root',type=Path)
    train.add_argument('--occlusion-index',type=int,choices=range(25))
    train.add_argument('--epochs',type=int,default=45)
    train.add_argument('--workers',type=int,default=16)
    train.add_argument('--seed',type=int,default=0)
    train.add_argument('--show-config',action='store_true',help='Print resolved Hydra settings without training')
    test=sub.add_parser('test',help='Evaluate the validation-ranked top five checkpoints in a training run')
    test.add_argument('--run-dir',type=Path,required=True)
    for q in [train,test]:
        q.add_argument('--stream',choices=['bilateral','left','right'],default='bilateral')
        q.add_argument('--dry-run',action='store_true',help='Print the command without executing it')
    a=p.parse_args(argv)
    folder='main_cls_code_dl' if a.stream=='bilateral' else 'main_cls_code_dl_ablation_single_stream'
    env=os.environ.copy()
    if a.action=='train':
        if (a.occlusion_index is None)!=(a.occlusion_root is None):p.error('Set both --occlusion-index and --occlusion-root for a structure experiment')
        model=a.model if a.stream=='bilateral' else 'ablation_'+a.stream+'_eye_only_tinynet_c'
        if not (ROOT/folder/'conf/model'/(model+'.yaml')).is_file():p.error('Unknown model configuration: '+model)
        env.update(PWV_IMAGE_ROOT=str(a.image_root.resolve()),PWV_SPLIT_ROOT=str(a.split_root.resolve()))
        cmd=[sys.executable,str(ROOT/folder/'train_main.py'),'model='+model,'hydra.run.dir='+str(a.output_dir.resolve()),'max_epochs='+str(a.epochs),'num_workers='+str(a.workers),'seed='+str(a.seed)]
        if a.occlusion_index is not None:
            env['PWV_OCCLUSION_ROOT']=str(a.occlusion_root.resolve());cmd+=['occlusion_seg_index='+str(a.occlusion_index)]
        if a.show_config:cmd+=['--cfg','job','--resolve']
        if not (a.dry_run or a.show_config) and a.output_dir.exists() and any(a.output_dir.iterdir()):p.error('Choose a new output directory for this training run')
    else:
        cmd=[sys.executable,str(ROOT/folder/'test_main.py'),str(a.run_dir.resolve()),'0']
    if a.dry_run:
        print(json.dumps({'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith('PWV_')}},indent=2));return 0
    return subprocess.run(cmd,cwd=ROOT,env=env).returncode
if __name__=='__main__':raise SystemExit(main())
