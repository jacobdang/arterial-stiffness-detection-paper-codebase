#!/usr/bin/env python3
"""Render study figures from released aggregate results without fitting models."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures',nargs='+',choices=['1','2A','4','5','6','S1','S4','S7','S8'],default=['2A','4','5','6','S1','S4','S7','S8'])
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--roc-coordinates',type=Path,help='Figure 1: cohort, model, false_positive_rate, true_positive_rate')
    a=p.parse_args()
    import pandas as pd
    import matplotlib.pyplot as plt
    import study_plots as plots
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name in a.figures:
        out=a.output_dir/('Figure_'+name)
        if name=='1':
            if a.roc_coordinates is None:p.error('Figure 1 requires --roc-coordinates from authorized participant predictions')
            metrics=pd.read_csv(ROOT/'results/figure_source_data/figure1_auc_metrics.csv')
            plots.plot_figure1(metrics,pd.read_csv(a.roc_coordinates),out)
        elif name=='2A':
            df=pd.read_csv(ROOT/'results/shap_20mi/shap_feature_importance_20mi.csv').sort_values('pooled_mean_abs_shap')
            fig,ax=plt.subplots(figsize=(8,5))
            ax.barh(df.feature_name,df.pooled_mean_abs_shap,xerr=df.between_imputation_sd,capsize=2)
            ax.set_xlabel('Mean absolute SHAP value (log-odds), averaged across 20 imputations')
            fig.tight_layout();plots.save_figure(fig,out)
        elif name=='4':
            df=pd.read_csv(ROOT/'results/figure_source_data/figure4_occlusion_metrics.csv')
            metrics={r.analysis:{'n':r.n,'positive':r.events,'auc':r.auc,'auc_ci_lower':r.ci_lower,'auc_ci_upper':r.ci_upper,'bootstrap_resamples':r.bootstrap_resamples,'bootstrap_seed':r.bootstrap_seed} for r in df.itertuples()}
            reference=pd.read_csv(ROOT/'results/figure_source_data/figure1_auc_metrics.csv')
            full=reference.loc[(reference.cohort=='internal_test') & (reference.model=='Image model')].iloc[0]
            metrics['full_image']={'auc':float(full.auc)}
            plots.plot_figure4(metrics,out)
        elif name=='5':
            from plot_figure5 import load_frozen_rows,render
            rows,_=load_frozen_rows(ROOT/'results/figure5/figure5_associations.csv')
            folder=a.output_dir/'Figure_5';folder.mkdir(exist_ok=True);render(rows,folder)
        elif name=='6':plots.plot_figure6(out)
        elif name=='S7':
            data=json.loads((ROOT/'results/clinical_utility/screening_costs.json').read_text())
            plots.plot_supplement_s7(data,out)
        else:getattr(plots,'plot_supplement_'+name.lower())(out)
    print('Rendered: '+', '.join(a.figures))
if __name__=='__main__':main()
