
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from scipy.stats import norm
import sklearn.metrics as metrics
import sklearn.metrics as sklm

# ---------- Core helpers ----------

def _as_binary01(y_true):
    y = np.asarray(y_true).astype(int)
    u = np.unique(y)
    if set(u.tolist()) == {-1, 1}:
        y = (y == 1).astype(int)
        u = np.unique(y)
    if not np.array_equal(u, [0, 1]) and not np.array_equal(u, [0]) and not np.array_equal(u, [1]):
        raise ValueError(f"y_true must be binary 0/1. Got unique={u}")
    return y

def cal_sep(test_labs_at_dim, rounded_preds_at_dim):
    cm = sklm.confusion_matrix(test_labs_at_dim, rounded_preds_at_dim)
    if len(cm)==2:
        specificity =cm[0,0]/(cm[0,0]+cm[0,1])
    else:
        specificity = 0
    return specificity

def format_pvalue(p, digits=3):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    p = float(p)
    if p <= 0:
        return "<1e-323"
    if p < 10**(-(digits+1)):
        return f"{p:.2e}"
    return f"{p:.{digits}f}"

def percentile_ci(values, alpha=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    lo = (1 - alpha) / 2 * 100
    hi = (1 + alpha) / 2 * 100
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))

def percentile_ci_index(values):
    """
    95% CI using the same integer-index method as stats_fun.py cal_ci95:
      sorted_scores[int(0.025 * len(sorted_scores))]
      sorted_scores[int(0.975 * len(sorted_scores))]
    No NaN filtering, no interpolation.
    """
    sorted_scores = np.array(values, dtype=float)
    sorted_scores.sort()
    lo = sorted_scores[int(0.025 * len(sorted_scores))]
    hi = sorted_scores[int(0.975 * len(sorted_scores))]
    return float(lo), float(hi)

def assert_all_equal(arr_list, name="array"):
    arr0 = np.asarray(arr_list[0])
    for i, arr in enumerate(arr_list[1:], start=1):
        if not np.array_equal(arr0, np.asarray(arr)):
            raise AssertionError(f"{name}[0] and {name}[{i}] are not identical")

# ---------- Original-style operating points ----------

def youden_operating_point(y_true, y_pred):
    y = _as_binary01(y_true)
    p = np.asarray(y_pred, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    sep = 1 - fpr
    idx = np.argmax(tpr + sep - 1)
    return float(tpr[idx]), float(sep[idx]), float(thresholds[idx])

def specificity_at_fixed_recall(y_true, y_pred, recall_target):
    y = _as_binary01(y_true)
    p = np.asarray(y_pred, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    sep = 1 - fpr
    idx = np.argmin(np.abs(tpr - recall_target))
    return float(tpr[idx]), float(sep[idx]), float(thresholds[idx])

def bootstrap_sens_spec_at_threshold(y_true, y_pred, threshold, n_bootstraps=1000, seed=42):
    y = _as_binary01(y_true)
    p = np.asarray(y_pred, dtype=float)
    rng = np.random.RandomState(seed)
    sens_vals, spec_vals = [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        yb = y[idx]
        pb = p[idx]
        pred_bin = (pb >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, pred_bin, labels=[0,1]).ravel()
        sens_vals.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        spec_vals.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    sens_vals = np.asarray(sens_vals, dtype=float)
    spec_vals = np.asarray(spec_vals, dtype=float)
    sens_vals = sens_vals[~np.isnan(sens_vals)]
    spec_vals = spec_vals[~np.isnan(spec_vals)]
    return (
        float(np.mean(sens_vals)),
        *percentile_ci_index(sens_vals)
    ), (
        float(np.mean(spec_vals)),
        *percentile_ci_index(spec_vals)
    )

def recall_and_sep_and_th_largest_youden(y_test, y_pred_val):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_val)
    sep = 1 - fpr
    best_thre_index = np.argmax(tpr + sep - 1)
    best_thre = thresholds[best_thre_index]
    best_sep = sep[best_thre_index]
    best_recall = tpr[best_thre_index]
    return best_recall, best_sep, best_thre

def sep_and_th_for_a_given_recall(y_test, y_pred_val, recall_thre):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_val)
    recall = tpr
    sep = 1 - fpr
    best_thre_index = np.argmin(np.abs(recall - recall_thre))
    best_thre = thresholds[best_thre_index]
    best_sep = sep[best_thre_index]
    result = ((y_pred_val <best_thre) & (y_test==0))
    return recall[best_thre_index], best_sep, best_thre

def cal_ci95_fixed(y_true,y_pred,recall_target=0.8):
    th = recall_target
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    global_auc = roc_auc_score(y_true, y_pred)
    global_largest_youden_rec, global_largest_youden_sep, global_largest_youden_th = recall_and_sep_and_th_largest_youden(y_true, y_pred)
    global_given_recall_rec, global_given_recall_sep, global_given_recall_th = sep_and_th_for_a_given_recall(y_true, y_pred, th)
    
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_auc = []
    bootstrapped_largest_youden_recall = []
    bootstrapped_largest_youden_sep = []
    bootstrapped_given_recall_sep = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score_auc = roc_auc_score(y_true[indices], y_pred[indices])
        recall = sklm.recall_score(y_true[indices], y_pred[indices]>=global_largest_youden_th, average=None)
        recall = recall[1]
        sep = cal_sep(y_true[indices], y_pred[indices]>=global_largest_youden_th)
        sep_th = cal_sep(y_true[indices], y_pred[indices]>=global_given_recall_th)
        bootstrapped_auc.append(score_auc)
        bootstrapped_largest_youden_recall.append(recall)
        bootstrapped_largest_youden_sep.append(sep)
        bootstrapped_given_recall_sep.append(sep_th)

    auc=[]
    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    auc.append(sorted_scores[int(0.025 * len(sorted_scores))])
    auc.append(sorted_scores[int(0.975 * len(sorted_scores))])
    
    rec=[]
    sorted_scores = np.array(bootstrapped_largest_youden_recall)
    sorted_scores.sort()
    rec.append(sorted_scores[int(0.025 * len(sorted_scores))])
    rec.append(sorted_scores[int(0.975 * len(sorted_scores))])
    
    sep = []
    sorted_scores = np.array(bootstrapped_largest_youden_sep)
    sorted_scores.sort()
    sep.append(sorted_scores[int(0.025 * len(sorted_scores))])
    sep.append(sorted_scores[int(0.975 * len(sorted_scores))])
    
    sep_th = []
    sorted_scores = np.array(bootstrapped_given_recall_sep)
    sorted_scores.sort()
    sep_th.append(sorted_scores[int(0.025 * len(sorted_scores))])
    sep_th.append(sorted_scores[int(0.975 * len(sorted_scores))])
    
    print(auc)
    print(global_auc)
    return auc, rec, sep, sep_th, global_auc, global_largest_youden_rec, global_largest_youden_sep, global_given_recall_sep, global_largest_youden_th, global_given_recall_th
    


# ---------- Bootstrap metrics ----------

def auc_ci95_bootstrap(y_true, y_pred, n_bootstraps=2000, seed=42):
    y = _as_binary01(y_true)
    p = np.asarray(y_pred, dtype=float)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    point = float(roc_auc_score(y, p))
    lo, hi = percentile_ci_index(vals)
    return point, lo, hi

def auprc_ci95_bootstrap(y_true, y_pred, n_bootstraps=2000, seed=42):
    y = _as_binary01(y_true)
    p = np.asarray(y_pred, dtype=float)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], p[idx]))
    point = float(average_precision_score(y, p))
    lo, hi = percentile_ci_index(vals)
    return point, lo, hi

def brier_ci95_bootstrap(y_true, y_pred, n_bootstraps=2000, seed=42):
    y = _as_binary01(y_true)
    p = np.asarray(y_pred, dtype=float)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y), len(y))
        vals.append(brier_score_loss(y[idx], p[idx]))
    point = float(brier_score_loss(y, p))
    lo, hi = percentile_ci_index(vals)
    return point, lo, hi

def calibration_slope_intercept_ci95(y_true, y_pred, n_bootstraps=2000, seed=42):
    y = _as_binary01(y_true)
    p = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1-1e-6)
    x = logit(p).reshape(-1,1)
    rng = np.random.RandomState(seed)
    slopes, intercepts = [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        clf = LogisticRegression(solver='lbfgs', max_iter=200)
        clf.fit(x[idx], y[idx])
        slopes.append(float(clf.coef_[0][0]))
        intercepts.append(float(clf.intercept_[0]))
    return {
        "slope": (float(np.mean(slopes)), *percentile_ci_index(slopes)),
        "intercept": (float(np.mean(intercepts)), *percentile_ci_index(intercepts))
    }

# ---------- Reclassification metrics ----------

def continuous_nri(y_true, p_old, p_new):
    """
    Category-free / continuous NRI (Pencina definition):
      NRI = [P(new>old|event)-P(new<old|event)] + [P(new<old|nonevent)-P(new>old|nonevent)]
    """
    y = _as_binary01(y_true)
    old = np.asarray(p_old, dtype=float)
    new = np.asarray(p_new, dtype=float)
    ev = (y == 1)
    ne = (y == 0)
    if ev.sum() == 0 or ne.sum() == 0:
        return np.nan, np.nan, np.nan
    u_ev = np.where(new[ev] > old[ev], 1.0, np.where(new[ev] < old[ev], -1.0, 0.0))
    u_ne = np.where(new[ne] < old[ne], 1.0, np.where(new[ne] > old[ne], -1.0, 0.0))
    nri_ev = float(u_ev.mean())
    nri_ne = float(u_ne.mean())
    nri = float(nri_ev + nri_ne)
    return nri, nri_ev, nri_ne

def cal_continuous_nri_ci95(y_true, p_old, p_new, ci_method="asymptotic", n_bootstraps=2000, seed=42):
    """
    PredictABEL/Pencina-style continuous NRI for binary outcomes:
    - point estimate as category-free NRI
    - asymptotic z-test p-value
    - CI by asymptotic or percentile bootstrap
    """
    y = _as_binary01(y_true)
    old = np.asarray(p_old, dtype=float)
    new = np.asarray(p_new, dtype=float)
    ev = (y == 1)
    ne = (y == 0)
    n_ev, n_ne = int(ev.sum()), int(ne.sum())
    if n_ev == 0 or n_ne == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    nri, nri_ev, nri_ne = continuous_nri(y, old, new)

    u_ev = np.where(new[ev] > old[ev], 1.0, np.where(new[ev] < old[ev], -1.0, 0.0))
    u_ne = np.where(new[ne] < old[ne], 1.0, np.where(new[ne] > old[ne], -1.0, 0.0))
    var_ev = (np.mean(u_ev**2) - nri_ev**2) / n_ev
    var_ne = (np.mean(u_ne**2) - nri_ne**2) / n_ne
    se = float(np.sqrt(max(var_ev + var_ne, 0.0)))
    if se > 0:
        z = float(nri / se)
        p = float(2.0 * norm.sf(abs(z)))
        ci_low_a = float(nri - 1.96 * se)
        ci_high_a = float(nri + 1.96 * se)
    else:
        z = np.nan
        p = np.nan
        ci_low_a = np.nan
        ci_high_a = np.nan

    if ci_method == "asymptotic":
        return nri, ci_low_a, ci_high_a, p, nri_ev, nri_ne, se, z

    if ci_method != "bootstrap":
        raise ValueError("ci_method must be 'asymptotic' or 'bootstrap'")

    rng = np.random.RandomState(seed)
    ev_idx = np.where(ev)[0]
    ne_idx = np.where(ne)[0]
    vals = []
    for _ in range(n_bootstraps):
        b_ev = rng.choice(ev_idx, size=n_ev, replace=True)
        b_ne = rng.choice(ne_idx, size=n_ne, replace=True)
        idx = np.concatenate([b_ev, b_ne])
        val, _, _ = continuous_nri(y[idx], old[idx], new[idx])
        vals.append(val)
    ci_low, ci_high = percentile_ci_index(vals)
    return nri, ci_low, ci_high, p, nri_ev, nri_ne, se, z

def idi(y_true, p_old, p_new):
    y = _as_binary01(y_true)
    old = np.asarray(p_old, dtype=float)
    new = np.asarray(p_new, dtype=float)
    delta = new - old
    ev = (y == 1)
    ne = (y == 0)
    if ev.sum() == 0 or ne.sum() == 0:
        return np.nan
    return float(delta[ev].mean() - delta[ne].mean())

def cal_idi_ci95(y_true, p_old, p_new, ci_method="asymptotic", n_bootstraps=2000, seed=42):
    """
    PredictABEL/Pencina-style IDI for binary outcomes:
      IDI = E(new-old | event) - E(new-old | nonevent)
    """
    y = _as_binary01(y_true)
    old = np.asarray(p_old, dtype=float)
    new = np.asarray(p_new, dtype=float)
    delta = new - old
    ev = (y == 1)
    ne = (y == 0)
    n_ev, n_ne = int(ev.sum()), int(ne.sum())
    if n_ev == 0 or n_ne == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    point = float(delta[ev].mean() - delta[ne].mean())
    var_ev = float(delta[ev].var(ddof=0) / n_ev)
    var_ne = float(delta[ne].var(ddof=0) / n_ne)
    se = float(np.sqrt(max(var_ev + var_ne, 0.0)))
    if se > 0:
        z = float(point / se)
        p = float(2.0 * norm.sf(abs(z)))
        ci_low_a = float(point - 1.96 * se)
        ci_high_a = float(point + 1.96 * se)
    else:
        z = np.nan
        p = np.nan
        ci_low_a = np.nan
        ci_high_a = np.nan

    if ci_method == "asymptotic":
        return point, ci_low_a, ci_high_a, p, se, z

    if ci_method != "bootstrap":
        raise ValueError("ci_method must be 'asymptotic' or 'bootstrap'")

    rng = np.random.RandomState(seed)
    ev_idx = np.where(ev)[0]
    ne_idx = np.where(ne)[0]
    vals = []
    for _ in range(n_bootstraps):
        b_ev = rng.choice(ev_idx, size=n_ev, replace=True)
        b_ne = rng.choice(ne_idx, size=n_ne, replace=True)
        idx = np.concatenate([b_ev, b_ne])
        vals.append(idi(y[idx], old[idx], new[idx]))
    ci_low, ci_high = percentile_ci_index(vals)
    return point, ci_low, ci_high, p, se, z

# ---------- Paired comparisons ----------

def paired_bootstrap_diff(y_true, p_old, p_new, metric_fn, n_bootstraps=2000, seed=42):
    """
    Bootstrap paired difference for metric_fn(y, p): metric(new) - metric(old)
    """
    y = _as_binary01(y_true)
    old = np.asarray(p_old, dtype=float)
    new = np.asarray(p_new, dtype=float)
    point_old = float(metric_fn(y, old))
    point_new = float(metric_fn(y, new))
    point_diff = point_new - point_old

    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(metric_fn(y[idx], new[idx]) - metric_fn(y[idx], old[idx])))
    ci_low, ci_high = percentile_ci_index(vals)
    p_two = float(min(1.0, 2.0 * min(np.mean(np.asarray(vals) <= 0.0), np.mean(np.asarray(vals) >= 0.0))))
    return point_old, point_new, point_diff, ci_low, ci_high, p_two

def delong_pvalue(y_true, p_old, p_new, delong_module=None):
    """
    roc_pvalue_python.delong_roc_test returns log10(p); convert to p.
    """
    y = _as_binary01(y_true)
    if delong_module is None:
        import roc_pvalue_python as delong_module
    log10p = delong_module.delong_roc_test(y.astype(int), np.asarray(p_old, dtype=float), np.asarray(p_new, dtype=float))
    return float(np.power(10.0, np.asarray(log10p)).ravel()[0])

# ---------- Table builders ----------

def _metric_summary(y_true, p, n_boot=2000, seed=42):
    # AUC, Sensitivity, Specificity: use cal_ci95_fixed (identical to stats_fun.cal_ci95)
    # cal_ci95_fixed always uses n_bootstraps=1000, seed=42 internally to match stats_fun
    auc_ci, rec_ci, sep_ci, sep_th_ci, global_auc, global_rec, global_sep, global_sep_th, youden_th, given_th = \
        cal_ci95_fixed(y_true, p, recall_target=0.8)

    # AUPRC, Brier, Calibration: separate bootstrap (these have no counterpart in stats_fun.cal_ci95)
    auprc_pt, auprc_lo, auprc_hi = auprc_ci95_bootstrap(y_true, p, n_bootstraps=n_boot, seed=seed+1000)
    brier_pt, brier_lo, brier_hi = brier_ci95_bootstrap(y_true, p, n_bootstraps=n_boot, seed=seed+2000)
    calib = calibration_slope_intercept_ci95(y_true, p, n_bootstraps=n_boot, seed=seed+3000)

    return {
        "auc": (global_auc, auc_ci[0], auc_ci[1]),
        "auprc": (auprc_pt, auprc_lo, auprc_hi),
        "brier": (brier_pt, brier_lo, brier_hi),
        "calib": calib,
        # sens: (point, ci_lo, ci_hi) — point is the global value, CI from bootstrap
        "sens": (global_rec, rec_ci[0], rec_ci[1]),
        # spec: (point, ci_lo, ci_hi) — point is the global value, CI from bootstrap
        "spec": (global_sep, sep_ci[0], sep_ci[1]),
    }

def run_table2_analysis(
    input_dir_single_var,
    indicator_name_list,
    get_score_and_gt_func,
    method_id_list=None,
    n_boot=2000,
    seed_base=100,
    output_table2_csv='table2.csv',
    output_pairwise_csv='pairwise.csv',
    factor_fusion_type='metric_only',
    fusion_fusion_type='fusion',
    feature_id_builder=None,
    delong_module=None,
    ci_method_reclass='asymptotic',
    include_diff_cols=False,
):
    if method_id_list is None:
        method_id_list = ['method_lr_basic']
    if feature_id_builder is None:
        feature_id_builder = lambda idx: "'" + str(idx) + "'"

    rows_table2 = []
    pairs = []

    for idx in range(1, len(indicator_name_list) + 1):
        name = indicator_name_list[idx - 1]
        feature_id = feature_id_builder(idx)
        print("Processing", idx, name)

        best_method, val_gt, test_gt, val_pred, test_pred = get_score_and_gt_func(
            input_dir_single_var, feature_id, fusion_type=factor_fusion_type, method_id_list=method_id_list
        )
        y_true = np.asarray(test_gt)
        p_factor = np.asarray(test_pred, dtype=float)

        best_method_f, val_gt_f, test_gt_f, val_pred_f, test_pred_f = get_score_and_gt_func(
            input_dir_single_var, feature_id, fusion_type=fusion_fusion_type, method_id_list=method_id_list
        )
        p_fusion = np.asarray(test_pred_f, dtype=float)

        if not np.array_equal(y_true, np.asarray(test_gt_f)):
            if len(y_true) != len(test_gt_f):
                raise ValueError(f"GT length mismatch for {name}")
            raise ValueError(f"GT order/content mismatch for {name}; align by sample id before analysis.")

        m_factor = _metric_summary(y_true, p_factor, n_boot=n_boot, seed=seed_base + idx)
        m_fusion = _metric_summary(y_true, p_fusion, n_boot=n_boot, seed=seed_base + 5000 + idx)

        # AUC paired comparison
        try:
            p_delong = delong_pvalue(y_true, p_factor, p_fusion, delong_module=delong_module)
        except Exception as ex:
            print(f"{name} DeLong error:", ex)
            p_delong = np.nan

        # Paired bootstrap P value for the AUPRC comparison
        auprc_old, auprc_new, auprc_diff, auprc_diff_lo, auprc_diff_hi, auprc_boot_p = paired_bootstrap_diff(
            y_true, p_factor, p_fusion, average_precision_score, n_bootstraps=n_boot, seed=seed_base + 900 + idx
        )

        # Reclassification metrics (continuous NRI + IDI only)
        nri_point, nri_lo, nri_hi, nri_p, _, _, _, _ = cal_continuous_nri_ci95(
            y_true, p_factor, p_fusion, ci_method=ci_method_reclass, n_bootstraps=n_boot, seed=seed_base + 1000 + idx
        )
        idi_point, idi_lo, idi_hi, idi_p, _, _ = cal_idi_ci95(
            y_true, p_factor, p_fusion, ci_method=ci_method_reclass, n_bootstraps=n_boot, seed=seed_base + 1100 + idx
        )

        rows_table2.append({
            'Indicator': name,
            'Model': 'Factor',
            'Sensitivity (95% CI)': f"{m_factor['sens'][0]*100:.1f} ({m_factor['sens'][1]*100:.1f}, {m_factor['sens'][2]*100:.1f})",
            'Specificity (95% CI)': f"{m_factor['spec'][0]*100:.1f} ({m_factor['spec'][1]*100:.1f}, {m_factor['spec'][2]*100:.1f})",
            'AUC (95% CI)': f"{m_factor['auc'][0]:.3f} ({m_factor['auc'][1]:.3f}, {m_factor['auc'][2]:.3f})",
            'AUPRC (95% CI)': f"{m_factor['auprc'][0]:.3f} ({m_factor['auprc'][1]:.3f}, {m_factor['auprc'][2]:.3f})",
            'Brier (95% CI)': f"{m_factor['brier'][0]:.4f} ({m_factor['brier'][1]:.4f}, {m_factor['brier'][2]:.4f})",
            'Calib slope (95% CI)': f"{m_factor['calib']['slope'][0]:.3f} ({m_factor['calib']['slope'][1]:.3f}, {m_factor['calib']['slope'][2]:.3f})",
            'Calib intercept (95% CI)': f"{m_factor['calib']['intercept'][0]:.3f} ({m_factor['calib']['intercept'][1]:.3f}, {m_factor['calib']['intercept'][2]:.3f})",
            'IDI (95% CI; p)': '-',
            'cNRI (95% CI; p)': '-',
            'AUC p (DeLong)': ''
        })
        rows_table2.append({
            'Indicator': name,
            'Model': 'Fusion',
            'Sensitivity (95% CI)': f"{m_fusion['sens'][0]*100:.1f} ({m_fusion['sens'][1]*100:.1f}, {m_fusion['sens'][2]*100:.1f})",
            'Specificity (95% CI)': f"{m_fusion['spec'][0]*100:.1f} ({m_fusion['spec'][1]*100:.1f}, {m_fusion['spec'][2]*100:.1f})",
            'AUC (95% CI)': f"{m_fusion['auc'][0]:.3f} ({m_fusion['auc'][1]:.3f}, {m_fusion['auc'][2]:.3f})",
            'AUPRC (95% CI)': f"{m_fusion['auprc'][0]:.3f} ({m_fusion['auprc'][1]:.3f}, {m_fusion['auprc'][2]:.3f})",
            'Brier (95% CI)': f"{m_fusion['brier'][0]:.4f} ({m_fusion['brier'][1]:.4f}, {m_fusion['brier'][2]:.4f})",
            'Calib slope (95% CI)': f"{m_fusion['calib']['slope'][0]:.3f} ({m_fusion['calib']['slope'][1]:.3f}, {m_fusion['calib']['slope'][2]:.3f})",
            'Calib intercept (95% CI)': f"{m_fusion['calib']['intercept'][0]:.3f} ({m_fusion['calib']['intercept'][1]:.3f}, {m_fusion['calib']['intercept'][2]:.3f})",
            'IDI (95% CI; p)': f"{idi_point:.4f} ({idi_lo:.4f}, {idi_hi:.4f}); p={format_pvalue(idi_p)}",
            'cNRI (95% CI; p)': f"{nri_point:.4f} ({nri_lo:.4f}, {nri_hi:.4f}); p={format_pvalue(nri_p)}",
            'AUC p (DeLong)': format_pvalue(p_delong)
        })

        pair_row = {
            'Indicator': name,
            'n_event': int(np.sum(np.asarray(y_true) == 1)),
            'n_nonevent': int(np.sum(np.asarray(y_true) == 0)),
            'AUC_factor': m_factor['auc'][0],
            'AUC_factor_CI_low': m_factor['auc'][1],
            'AUC_factor_CI_high': m_factor['auc'][2],
            'AUC_fusion': m_fusion['auc'][0],
            'AUC_fusion_CI_low': m_fusion['auc'][1],
            'AUC_fusion_CI_high': m_fusion['auc'][2],
            'AUC_p_delong': p_delong,
            'AUC_p_delong_str': format_pvalue(p_delong),
            'AUPRC_factor': m_factor['auprc'][0],
            'AUPRC_factor_CI_low': m_factor['auprc'][1],
            'AUPRC_factor_CI_high': m_factor['auprc'][2],
            'AUPRC_fusion': m_fusion['auprc'][0],
            'AUPRC_fusion_CI_low': m_fusion['auprc'][1],
            'AUPRC_fusion_CI_high': m_fusion['auprc'][2],
            'AUPRC_diff': auprc_diff,
            'AUPRC_diff_CI_low': auprc_diff_lo,
            'AUPRC_diff_CI_high': auprc_diff_hi,
            'AUPRC_boot_p': auprc_boot_p,
            'AUPRC_boot_p_str': format_pvalue(auprc_boot_p),
            'Brier_factor': m_factor['brier'][0],
            'Brier_factor_CI_low': m_factor['brier'][1],
            'Brier_factor_CI_high': m_factor['brier'][2],
            'Brier_fusion': m_fusion['brier'][0],
            'Brier_fusion_CI_low': m_fusion['brier'][1],
            'Brier_fusion_CI_high': m_fusion['brier'][2],
            'IDI': idi_point,
            'IDI_CI_low': idi_lo,
            'IDI_CI_high': idi_hi,
            'IDI_p': idi_p,
            'IDI_p_str': format_pvalue(idi_p),
            'cNRI': nri_point,
            'cNRI_CI_low': nri_lo,
            'cNRI_CI_high': nri_hi,
            'cNRI_p': nri_p,
            'cNRI_p_str': format_pvalue(nri_p),
        }
        if include_diff_cols:
            auc_old, auc_new, auc_diff, auc_diff_lo, auc_diff_hi, auc_boot_p = paired_bootstrap_diff(
                y_true, p_factor, p_fusion, roc_auc_score, n_bootstraps=n_boot, seed=seed_base + 800 + idx
            )
            pair_row.update({
                'AUC_diff': auc_diff,
                'AUC_diff_CI_low': auc_diff_lo,
                'AUC_diff_CI_high': auc_diff_hi,
                'AUC_boot_p': auc_boot_p,
                'AUC_boot_p_str': format_pvalue(auc_boot_p),
            })
        pairs.append(pair_row)

    df_table2 = pd.DataFrame(rows_table2)
    df_pairs = pd.DataFrame(pairs)
    df_table2.to_csv(output_table2_csv, index=False)
    df_pairs.to_csv(output_pairwise_csv, index=False)
    print(f"Saved {output_table2_csv} and {output_pairwise_csv}")
    return df_table2, df_pairs
