#!/usr/bin/env python3
"""Apply the 20 released fusion models to matching imputation-specific inputs."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-pattern',required=True,help='CSV path pattern containing {mi}, for imputations 1 through 20')
    p.add_argument('--id-column',default='participant_id')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    import pandas as pd
    from scipy.special import expit
    payload=json.loads((ROOT/'models/public_release/fusion_lr_20mi/fusion_lr_20mi_coefficients.json').read_text())
    names=payload['input_contract']['feature_order'];canonical=None;pred=[]
    for mi,model in enumerate(payload['models'],1):
        frame=pd.read_csv(a.input_pattern.format(mi=mi))
        if frame[a.id_column].isna().any() or frame[a.id_column].duplicated().any():p.error('Each imputation needs unique nonempty participant identifiers')
        frame=frame.set_index(a.id_column)
        if canonical is None:canonical=frame.index
        if set(frame.index)!=set(canonical):p.error('Imputation participant sets differ')
        x=frame.loc[canonical,names].to_numpy(dtype=float)
        if not np.isfinite(x).all():p.error('Encoded, imputed, normalized inputs must be finite')
        coef=np.array([entry['coefficient'] for entry in model['coefficients']])
        pred.append(expit(float(model['intercept'])+x@coef))
    result=pd.DataFrame({a.id_column:canonical,'fusion_probability':np.mean(pred,axis=0)})
    a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
if __name__=='__main__':main()
