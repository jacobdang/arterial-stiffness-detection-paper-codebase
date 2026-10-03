#!/usr/bin/env python3
"""Train and evaluate clinical/fusion models using the study's validation search."""
import argparse,json,logging,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--method',choices=['lr','lda','lsvm','rbfsvm','rf','gbc'],default='lr')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    import yaml
    c=yaml.safe_load(a.config.read_text())
    mi=c['mice_imputation_index_list'] if c['use_mice_imputation'] else [None]
    modes=['metric_only','fusion'] if c.get('fusion',True) else ['metric_only']
    if a.dry_run:
        print(json.dumps({'method':a.method,'imputations':mi,'modes':modes,'feature_sets':c['feat_set_list'],'output_dir':c['output_dir']},indent=2));return 0
    sys.path.insert(0,str(ROOT/'main_cls_code_metric_and_fusion'))
    import main_basic
    fit=getattr(main_basic,'train_and_test_model_main_'+a.method)
    base=Path(c['output_dir']).resolve()/('method_'+a.method+'_basic')
    base.mkdir(parents=True,exist_ok=False)
    for index in mi:
        for mode in modes:
            cfg=deepcopy(c)
            for k in ['train_csv_name','val_csv_name','test_csv_name']:
                cfg[k]=str(Path(cfg[k].replace('$X$',str(index))).resolve())
            for k in ['train_img_only_pickle_file','val_img_only_pickle_file','test_img_only_pickle_file']:
                if k in cfg:cfg[k]=str(Path(cfg[k]).resolve())
            out=base/('mice_imputation_'+str(index))/mode if index is not None else base/mode
            out.mkdir(parents=True);cfg['output_dir']=str(out)
            logger=logging.getLogger(str(out));logger.setLevel(logging.INFO)
            handle=logging.FileHandler(out/'train_and_test_main.log');logger.addHandler(handle)
            try:
                for features in cfg['feat_set_list']:
                    fit(logger,cfg,features,mode=='fusion',target=cfg['target'],threshold=cfg['threshold'])
            finally:logger.removeHandler(handle);handle.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
