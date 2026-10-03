#!/usr/bin/env python3
"""Calculate participant-level discrimination and paired model comparisons."""
import argparse,itertools,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--outcome',default='outcome')
    p.add_argument('--scores',nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--group',help='Optional subgroup column; analysis uses available values')
    a=p.parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score,brier_score_loss
    from metrics import auc_bootstrap_summary,independent_delong_test
    from roc_pvalue_python import delong_roc_test
    df=pd.read_csv(a.input)
    if 'participant_id' in df and df['participant_id'].duplicated().any():p.error('Use one row per participant')
    if not np.isfinite(df[[a.outcome,*a.scores]].to_numpy(dtype=float)).all():p.error('Outcomes and predictions must be finite')
    if not df[a.scores].apply(lambda x:x.between(0,1).all()).all():p.error('Prediction scores must be probabilities in [0, 1]')
    groups=list(df.dropna(subset=[a.group]).groupby(a.group)) if a.group else [('all',df)]
    rows=[];paired=[];between=[]
    for group,part in groups:
        y=part[a.outcome].to_numpy()
        for score in a.scores:
            pred=part[score].to_numpy()
            row=auc_bootstrap_summary(y,pred,a.bootstrap,a.seed)
            row.update(group=str(group),model=score,auprc=average_precision_score(y,pred),brier=brier_score_loss(y,pred))
            rows.append(row)
        for first,second in itertools.combinations(a.scores,2):
            logp=float(np.asarray(delong_roc_test(y,part[first].to_numpy(),part[second].to_numpy())).item())
            paired.append({'group':str(group),'first':first,'second':second,'p_value':10**logp})
    if a.group:
        for (g1,d1),(g2,d2) in itertools.combinations(groups,2):
            for score in a.scores:
                row=independent_delong_test(d1[a.outcome],d1[score],d2[a.outcome],d2[score])
                row.update(first_group=str(g1),second_group=str(g2),model=score);between.append(row)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output_dir/'performance.csv',index=False)
    if paired:pd.DataFrame(paired).to_csv(a.output_dir/'paired_delong.csv',index=False)
    if between:pd.DataFrame(between).to_csv(a.output_dir/'between_group_delong.csv',index=False)
    print(json.dumps({'models':len(a.scores),'groups':len(groups),'output_dir':str(a.output_dir)}))
if __name__=='__main__':main()
