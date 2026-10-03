#!/usr/bin/env python3
"""Run the 27 participant-mean vessel-feature correlations for Tables S11 and S12."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Participant/ROI means of vessel features with an aligned target')
    p.add_argument('--target',required=True,help='PWV or image-score column')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20260826)
    a=p.parse_args()
    import pandas as pd
    from vessel_correlations import _primary_participant_correlations
    df=pd.read_csv(a.input)
    if df.duplicated(['participant_id','roi']).any():p.error('Average available eyes within participant and ROI before analysis')
    df=df.copy();df['pwv']=df[a.target]
    result=_primary_participant_correlations(df,n_resamples=a.bootstrap,seed=a.seed)
    result['target']=a.target
    a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
if __name__=='__main__':main()
