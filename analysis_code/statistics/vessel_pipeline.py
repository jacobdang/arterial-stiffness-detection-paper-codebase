
"""
vessel_pipeline.py

Full pipeline:
 - robust vessel metric computations (raw vs. cleaned)
 - dataset-level calibration: calibrate_dataset_min_area
 - parallel dataset processing with joblib and tqdm

Designed for massive datasets (100k+ images) out-of-core via JSONL.
"""

from typing import Optional, Sequence, Tuple, Union, Callable, List, Dict, Any
import math
import warnings
import random
import os
import json

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import skimage.measure as measure

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    warnings.warn("tqdm not installed. Progress bars will not be shown.", UserWarning)

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    Parallel = None
    delayed = None
    _JOBLIB_AVAILABLE = False

try:
    import skan
    _SKAN_AVAILABLE = True
except ImportError:
    skan = None
    _SKAN_AVAILABLE = False
    warnings.warn("skan not available: graph-based metrics will be omitted.", UserWarning)

# CRITICAL FIX: Prevent OpenCV and Joblib thread collision deadlocks
cv2.setNumThreads(0)

# -------------------------
# --- Reproducibility
# -------------------------
def set_reproducibility(seed: int = 42):
    """Sets global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


ROI_MODES = {
    "full": None,
    "macula_dd1": "macula_map_dd1",
    "macula_dd2": "macula_map_dd2",
    "disc": "disc_map",
    "no_disc": "disc_map",   # handled specially
}

ROI_LIST = ["full", "disc", "macula_dd2"]


# -------------------------
# --- Core helpers
# -------------------------

_eps = 1e-8

def compute_central_light_reflex_from_image(img, vessel_mask, min_radius_px=1):
    """
    Compute central light reflex index for a vessel mask using the image.
    - img: HxWx3 (uint8) or HxW single-channel; will convert to grayscale
    - vessel_mask: binary uint8 mask of vessel pixels (same shape)
    Returns median_center / median_edge (or None if not computable).
    """
    if img is None or vessel_mask is None:
        return None

    # convert to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = gray.astype(np.float32)

    vessel_bool = (vessel_mask > 0)
    if vessel_bool.sum() == 0:
        return None

    # skeleton of the vessel: center pixels
    sk = skeletonize(vessel_bool).astype(bool)
    if sk.sum() == 0:
        # fallback: use medial axis by distance transform maxima
        if sk.sum() == 0:
            return None

    # Optionally filter skeleton pixels by radius to avoid tiny vessels
    dist = distance_transform_edt(vessel_bool)
    sk_filtered = sk & (dist >= min_radius_px)

    if sk_filtered.sum() < 5:
        # if too few center pixels with radius >= threshold, fallback to any skeleton pixels
        sk_filtered = sk

    if sk_filtered.sum() == 0:
        return None

    I_center = gray[sk_filtered]

    # vessel edge = vessel pixels excluding skeleton (interior/exterior)
    vessel_edge_mask = vessel_bool & (~sk)
    if vessel_edge_mask.sum() == 0:
        # fallback: dilate vessel and take ring outside vessel as edge
        k = _disk_kernel(1)
        if k is None:
            return None
        vessel_dil = cv2.dilate(vessel_mask.astype(np.uint8), k)
        ring = (vessel_dil > 0) & (~vessel_bool)
        if ring.sum() == 0:
            return None
        I_edge = gray[ring]
    else:
        I_edge = gray[vessel_edge_mask]

    # robust statistic: median ratio
    center_med = float(np.median(I_center)) if I_center.size > 0 else None
    edge_med = float(np.median(I_edge)) if I_edge.size > 0 else None

    if center_med is None or edge_med is None:
        return None

    return float(center_med / (edge_med + _eps))


def tortuosity_from_skan_stats(stats):
    if stats is None or len(stats) == 0:
        return None

    cols = stats.columns

    if "branch-distance" in cols and "euclidean-distance" in cols:
        geodesic = stats["branch-distance"].to_numpy(dtype=float)
        euclid   = stats["euclidean-distance"].to_numpy(dtype=float)
    else:
        return None

    valid = np.isfinite(geodesic) & np.isfinite(euclid) & (euclid > 0)
    if not np.any(valid):
        return None

    torts = geodesic[valid] / (euclid[valid] + _eps)
    torts = torts[np.isfinite(torts)]
    if torts.size == 0:
        return None

    return float(np.median(torts))



def _ensure_binary_uint8(mask):
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("Mask must be 2D")
    return (arr > 0).astype(np.uint8)

def _safe_percentiles(values, ps=(25, 50, 75, 90, 95)):
    if values is None or len(values) == 0:
        return {f"p{p}": None for p in ps}
    arr = np.asarray(values, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}

def _disk_kernel(radius: int):
    if radius <= 0:
        return None
    ksize = int(2 * radius + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

def morph_clean(mask: Optional[np.ndarray],
                morph_open_px: int = 0,
                morph_close_px: int = 0) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = (mask > 0).astype(np.uint8)
    
    if morph_open_px > 0:
        k = _disk_kernel(morph_open_px)
        if k is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            
    if morph_close_px > 0:
        k = _disk_kernel(morph_close_px)
        if k is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            
    return m

def prune_label_image_by_area(lbl: np.ndarray, min_area_px: int):
    if lbl is None:
        return None
    lbl = lbl.astype(int)
    maxlabel = int(lbl.max())
    if maxlabel == 0:
        return np.zeros_like(lbl, dtype=np.uint8)
    counts = np.bincount(lbl.ravel(), minlength=(maxlabel + 1))
    valid = np.nonzero(counts >= int(max(1, min_area_px)))[0]
    valid = valid[valid != 0]
    if valid.size == 0:
        return np.zeros_like(lbl, dtype=np.uint8)
    keep_map = np.zeros(maxlabel + 1, dtype=np.uint8)
    keep_map[valid] = 1
    return keep_map[lbl]

def widths_from_skeleton(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return np.array([], dtype=float)
    dist = distance_transform_edt(mask_u8)
    sk = skeletonize(mask_u8).astype(bool)
    return 2.0 * dist[sk]

def box_counting_fractal_dimension(mask):
    Z = (mask > 0).astype(np.uint8)
    if Z.sum() == 0:
        return None
    max_dim = max(Z.shape)
    n = 2 ** int(np.ceil(np.log2(max_dim)))
    if n < 8:
        return None
        
    pad_y = n - Z.shape[0]
    pad_x = n - Z.shape[1]
    
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before
    pad_x_before = pad_x // 2
    pad_x_after = pad_x - pad_x_before
    
    Zc = np.pad(Z, ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='constant', constant_values=0)
    
    max_k = int(np.log2(n))
    sizes = 2 ** np.arange(max_k, 0, -1)
    counts = []
    
    for size in sizes:
        S = np.add.reduceat(np.add.reduceat(Zc, np.arange(0, n, size), axis=0),
                            np.arange(0, n, size), axis=1)
        counts.append(np.sum(S > 0))
        
    counts = np.array(counts, dtype=float)
    nonzero = counts > 0
    if np.sum(nonzero) < 2:
        return None
        
    sizes_nz = sizes[nonzero]
    counts_nz = counts[nonzero]
    try:
        coeffs = np.polyfit(np.log(sizes_nz), np.log(counts_nz), 1)
    except Exception:
        return None
    if np.isnan(coeffs[0]):
        return None
    return float(-coeffs[0])

def skan_graph_stats(mask, spacing: Union[float, Tuple[float, float]] = 1.0, compute_graph: bool = True):
    if mask is None or mask.sum() == 0:
        return None, None
    skel_bool = skeletonize((mask > 0).astype(np.uint8)).astype(bool)
    skel_uint8 = skel_bool.astype(np.uint8)
    if (not _SKAN_AVAILABLE) or (not compute_graph):
        return None, skel_uint8
    try:
        spac = (float(spacing[0]), float(spacing[1])) if isinstance(spacing, (list, tuple)) else (float(spacing), float(spacing))
        sk = skan.Skeleton(skel_bool, spacing=spac)
        return skan.summarize(sk), skel_uint8
    except Exception:
        return None, skel_uint8

def graph_length_from_skan_stats(stats):
    if stats is None or len(stats) == 0:
        return 0.0
    cols = list(stats.columns)
    candidates = [c for c in cols if 'branch' in c and 'distance' in c] or [c for c in cols if c in ('branch-distance', 'branch_distance')]
    if candidates:
        try:
            return float(np.nansum(stats[candidates[0]].values.astype(float)))
        except Exception:
            return 0.0
    return 0.0

def branch_endpoint_node_counts_from_stats(stats):
    if stats is None or len(stats) == 0:
        return 0, 0
    
    cols = list(stats.columns)
    
    src_col = next((c for c in cols if c in ('node-id-src', 'node_id_src')), None)
    dst_col = next((c for c in cols if c in ('node-id-dst', 'node_id_dst')), None)
    
    if src_col and dst_col:
        nodes = np.concatenate([stats[src_col].values, stats[dst_col].values])
        _, counts = np.unique(nodes, return_counts=True)
        return int(np.sum(counts == 1)), int(np.sum(counts > 2))
        
    def find_col(suffixes):
        return next((c for s in suffixes for c in cols if c.endswith(s)), None)
    src0, src1 = find_col(['src-0', 'src_0']), find_col(['src-1', 'src_1'])
    dst0, dst1 = find_col(['dst-0', 'dst_0']), find_col(['dst-1', 'dst_1'])
    
    if None in (src0, src1, dst0, dst1):
        return 0, 0
    try:
        src = np.vstack([stats[src0].values, stats[src1].values]).T.astype(float)
        dst = np.vstack([stats[dst0].values, stats[dst1].values]).T.astype(float)
        coords = np.vstack([src, dst])
        _, counts = np.unique(np.round(coords).astype(int), axis=0, return_counts=True)
        return int(np.sum(counts == 1)), int(np.sum(counts > 2))
    except Exception:
        return 0, 0

def auto_min_component_area_px(mask: np.ndarray, shape: Tuple[int, int], fallback_px: int = 25) -> int:
    """
    Dynamically estimates a minimum component area for noise removal on a single image.
    Uses a heuristic based on image resolution and median skeleton width.
    """
    if mask is None or mask.sum() == 0:
        return fallback_px

    H, W = shape
    total_pixels = H * W
    area_res = total_pixels * 0.00005  # Base threshold on a small fraction of total resolution

    widths = widths_from_skeleton(mask)
    if widths.size > 0:
        med_w = np.median(widths)
        area_width = med_w ** 2  # Approximate noise artifact size as a circle bounded by vessel width
    else:
        area_width = 0

    suggested = int(round(max(area_res, area_width, fallback_px)))
    return suggested

# -------------------------
# --- Metric Computation
# -------------------------
def compute_vessel_metrics(artery_mask,
                           vein_mask,
                           pixel_size_mm: Optional[Union[float, Tuple[float, float]]] = None,
                           min_component_area_px: Optional[int] = None,
                           auto_min_area: bool = True,
                           fallback_min_area_px: int = 25,
                           morph_open_px: int = 0,
                           morph_close_px: int = 0,
                           compute_graph: bool = True,
                           compute_fractal: bool = True):
    
    if artery_mask is None or vein_mask is None:
        raise ValueError("artery_mask and vein_mask must be provided.")
    
    # 1. Base initialization
    art_raw = _ensure_binary_uint8(artery_mask)
    vein_raw = _ensure_binary_uint8(vein_mask)
    combined_raw = ((art_raw > 0) | (vein_raw > 0)).astype(np.uint8)
    
    H, W = art_raw.shape
    image_area = int(H * W)

    # 2. Morphological Cleaning
    art_morph = morph_clean(art_raw, morph_open_px, morph_close_px)
    vein_morph = morph_clean(vein_raw, morph_open_px, morph_close_px)

    # 3. Dynamic Area Calculation
    if auto_min_area and (min_component_area_px is None or min_component_area_px <= 0):
        combined_for_est = ((art_morph > 0) | (vein_morph > 0)).astype(np.uint8)
        min_area_to_use = auto_min_component_area_px(combined_for_est, combined_for_est.shape, fallback_px=fallback_min_area_px)
    else:
        min_area_to_use = max(1, int(min_component_area_px if min_component_area_px else fallback_min_area_px))

    # 4. Final Pruning
    combined_morph = ((art_morph > 0) | (vein_morph > 0)).astype(np.uint8)
    lbl_combined = measure.label(combined_morph, connectivity=2)
    valid_global_mask = prune_label_image_by_area(lbl_combined, min_area_to_use)

    art_clean = (art_morph > 0).astype(np.uint8) & valid_global_mask
    vein_clean = (vein_morph > 0).astype(np.uint8) & valid_global_mask
    combined_clean = ((art_clean > 0) | (vein_clean > 0)).astype(np.uint8)

    # 5. Analysis Helper
    spacing_for_skan = float(pixel_size_mm) if isinstance(pixel_size_mm, (int, float)) else (
        (float(pixel_size_mm[0]), float(pixel_size_mm[1])) if pixel_size_mm else 1.0)

    def analyze_mask(mask: Optional[np.ndarray], roi_area_px: int):
        out = {}
        area_px = int((mask > 0).astype(np.uint8).sum()) if mask is not None else 0
        out['area_px'] = area_px
        out['area_frac'] = float(area_px / roi_area_px) if roi_area_px > 0 else None

        if mask is not None and area_px > 0:
            lbl_local = measure.label(mask, connectivity=2)
            out['num_components'] = int(lbl_local.max())
        else:
            out['num_components'] = 0

        widths = widths_from_skeleton(mask) if mask is not None else np.array([], dtype=float)
        out['width_count'] = int(widths.size)
        if widths.size == 0:
            out.update({'width_mean_px': None, 'width_median_px': None, **_safe_percentiles([])})
        else:
            out['width_mean_px'] = float(np.mean(widths))
            out['width_median_px'] = float(np.median(widths))
            out.update(_safe_percentiles(widths))

        # Warning: Computing graph stats on heavily noisy raw masks can be slow
        stats, skel = skan_graph_stats(mask, spacing=spacing_for_skan, compute_graph=compute_graph)
        
        out['skeleton_pixel_count'] = int(np.count_nonzero(skel)) if skel is not None else 0
        out['length_px'] = float(graph_length_from_skan_stats(stats)) if compute_graph else None

        # Guard against silent skan schema mismatch
        if compute_graph and skel is not None and skel.sum() > 0 and out['length_px'] == 0:
            out['length_px'] = None

        ep, jp = branch_endpoint_node_counts_from_stats(stats) if compute_graph else (0, 0)
        out['endpoints'], out['junctions'] = int(ep), int(jp)

        out['fractal_dimension'] = box_counting_fractal_dimension(mask) if compute_fractal else None
        
        return out

    # 6. Structured Output Generation
    res = {
        'params': {
            'min_component_area_px': int(min_area_to_use),
            'auto_min_area_used': auto_min_area
        },
        'overlap_px_raw': int(((art_raw > 0) & (vein_raw > 0)).sum()),
        'overlap_px_cleaned': int(((art_clean > 0) & (vein_clean > 0)).sum()),
        
        # Raw Metrics (Before noise removal)
        'overall_raw': analyze_mask(combined_raw, image_area),
        'artery_raw': analyze_mask(art_raw, image_area),
        'vein_raw': analyze_mask(vein_raw, image_area),
        
        # Cleaned Metrics (After area & morph filtering)
        'overall_cleaned': analyze_mask(combined_clean, image_area),
        'artery_cleaned': analyze_mask(art_clean, image_area),
        'vein_cleaned': analyze_mask(vein_clean, image_area)
    }
    
    return res

def apply_roi(artery, vein, item, roi_mode):
    """
    artery, vein: uint8 vessel maps
    item: per-image dict containing ROI maps
    roi_mode: one of ROI_MODES keys
    """
    if roi_mode == "full" or roi_mode is None:
        return artery, vein

    if roi_mode == "no_disc":
        disc = item["disc_map"].astype(np.uint8)
        return artery * (1 - disc), vein * (1 - disc)

    roi_key = ROI_MODES.get(roi_mode)
    if roi_key is None:
        raise ValueError(f"Unknown ROI mode: {roi_mode}")

    roi = item[roi_key].astype(np.uint8)
    return artery * roi, vein * roi

# -------------------------
# --- Calibration utility 
# -------------------------
def calibrate_dataset_min_area_from_maps(
    items,                  # list of loaded dicts
    sample_size=500,
    random_seed=42,
    fallback_px=25,
    roi_key=None             # None / 'macula_map_dd2' / etc.
):
    set_reproducibility(random_seed)

    N = len(items)
    if N == 0:
        raise ValueError("Empty dataset")

    sample_n = min(sample_size, N)
    indices = random.sample(range(N), sample_n)

    w_meds, w_p25s, comp_areas_all = [], [], []
    H = W = None

    iterator = tqdm(indices, desc="Calibrating") if _TQDM_AVAILABLE else indices

    for idx in iterator:
        d = items[idx]
        if not d.get("all_valid", True):
            continue

        artery = d["artery_map"].astype(np.uint8)
        vein   = d["vein_map"].astype(np.uint8)

        if roi_key is not None:
            roi = d[roi_key].astype(np.uint8)
            artery = artery * roi
            vein   = vein * roi

        combined = ((artery > 0) | (vein > 0)).astype(np.uint8)
        if combined.sum() == 0:
            continue

        if H is None:
            H, W = combined.shape

        widths = widths_from_skeleton(combined)
        if widths.size < 10:
            continue

        w_meds.append(np.median(widths))
        w_p25s.append(np.percentile(widths, 25))

        areas = np.bincount(
            measure.label(combined, connectivity=2).ravel()
        )[1:]
        if areas.size > 0:
            comp_areas_all.append(areas)

    if not w_meds:
        return {"min_component_area_px": fallback_px}

    all_areas = np.concatenate(comp_areas_all) if comp_areas_all else np.array([], dtype=float)
    if all_areas.size == 0:
        return {"min_component_area_px": fallback_px}
    A_min_true = np.percentile(all_areas, 5)

    alpha_max = A_min_true / (np.median(w_meds) ** 2)
    beta_max  = A_min_true / (np.median(w_p25s) ** 2)

    alpha = np.clip(alpha_max * 0.9, 0.8, 1.6)
    beta  = np.clip(beta_max  * 0.9, 0.4, 0.9)
    gamma = 2.25

    area_width = max(
        alpha * np.median(w_meds) ** 2,
        beta  * np.median(w_p25s) ** 2
    )
    area_res = gamma * (H * W / 1e6)

    return {
        "min_component_area_px": int(round(max(area_width, area_res, fallback_px)))
    }

# -------------------------
# --- Dataset processing: Disk-streaming optimized
# -------------------------
def process_single_item_from_maps(
    item,
    compute_args,
    roi_modes=("full", "disc", "macula_dd2"),
    fast_mode=False
):
    if not item.get("all_valid", True):
        return None

    # raw maps
    base_artery = item["artery_map"].astype(np.uint8)
    base_vein   = item["vein_map"].astype(np.uint8)

    results = []

    for roi_mode in roi_modes:
        # apply ROI
        artery, vein = apply_roi(base_artery.copy(), base_vein.copy(), item, roi_mode)
    
        if artery.sum() == 0 and vein.sum() == 0:
            continue
            
        # prepare compute args copy
        c_args = dict(compute_args)
        if fast_mode:
            c_args["compute_graph"] = False
            c_args["compute_fractal"] = False
    
        # main metrics (unchanged)
        res = compute_vessel_metrics(artery, vein, **c_args)
        res["id"] = item.get("id", "unknown")
        res["roi_mode"] = roi_mode
    
        # replicate cleaning logic to obtain cleaned masks (so CLR/tortuosity are computed on the same masks)
        morph_open_px = int(c_args.get("morph_open_px", 0))
        morph_close_px = int(c_args.get("morph_close_px", 0))
        auto_min_area = bool(c_args.get("auto_min_area", True))
        fallback_min_area_px = int(c_args.get("fallback_min_area_px", 25))
        min_component_area_px = c_args.get("min_component_area_px", None)
    
        art_morph = morph_clean(artery, morph_open_px, morph_close_px)
        vein_morph = morph_clean(vein, morph_open_px, morph_close_px)
    
        if auto_min_area and (min_component_area_px is None or int(min_component_area_px) <= 0):
            combined_for_est = ((art_morph > 0) | (vein_morph > 0)).astype(np.uint8)
            min_area_to_use = auto_min_component_area_px(combined_for_est, combined_for_est.shape, fallback_px=fallback_min_area_px)
        else:
            min_area_to_use = max(1, int(min_component_area_px if min_component_area_px else fallback_min_area_px))
    
        combined_morph = ((art_morph > 0) | (vein_morph > 0)).astype(np.uint8)
        lbl_combined = measure.label(combined_morph, connectivity=2)
        valid_global_mask = prune_label_image_by_area(lbl_combined, int(min_area_to_use))
    
        art_clean = (art_morph > 0).astype(np.uint8) & valid_global_mask
        vein_clean = (vein_morph > 0).astype(np.uint8) & valid_global_mask
        combined_clean = ((art_clean > 0) | (vein_clean > 0)).astype(np.uint8)
    
        # ---- Additional features: CLR and tortuosity ----
        # Central Light Reflex: for arteries and veins (pixel-domain ratio)
        img = item.get("img", None)  # expect HxWx3 or HxW
        try:
            if img is not None:
                # artery CLR (use artery cleaned mask)
                artery_clr = compute_central_light_reflex_from_image(img, art_clean, min_radius_px=1)
                vein_clr = compute_central_light_reflex_from_image(img, vein_clean, min_radius_px=1)
            else:
                artery_clr = None
                vein_clr = None
        except Exception:
            artery_clr = None
            vein_clr = None
    
        res["artery_clr"] = artery_clr
        res["vein_clr"] = vein_clr
    
        # Tortuosity: use skan summary stats if possible
        try:
            tort_art = None
            tort_vein = None
            if _SKAN_AVAILABLE and c_args.get("compute_graph", True):
                # create skeleton and stats for artery_clean and vein_clean
                try:
                    stats_art, skel = skan_graph_stats(art_clean, spacing=1.0, compute_graph=True)
                    stats_vein, _ = skan_graph_stats(vein_clean, spacing=1.0, compute_graph=True)
                    tort_art = tortuosity_from_skan_stats(stats_art)
                    tort_vein = tortuosity_from_skan_stats(stats_vein)
                except Exception:
                    tort_art = None
                    tort_vein = None
            else:
                tort_art = None
                tort_vein = None
        except Exception:
            tort_art = None
            tort_vein = None
    
        res["artery_tortuosity_median_branch"] = tort_art
        res["vein_tortuosity_median_branch"] = tort_vein
        results.append(res)

    return results

def process_dataset_from_maps(
    items,
    output_jsonl,
    compute_args,
    roi_modes=("full", "disc", "macula_dd2"),
    n_jobs=-1,
    batch_size=500,
    fast_mode=True
):
    # ---- Safe directory creation ----
    out_dir = os.path.dirname(output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    iterator = range(0, len(items), batch_size)
    if _TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Processing (all ROIs)")

    for start in iterator:
        batch = items[start:start + batch_size]

        # ---- Parallel processing: each item returns a LIST of results ----
        if _JOBLIB_AVAILABLE:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_single_item_from_maps)(
                    item,
                    compute_args,
                    roi_modes=roi_modes,
                    fast_mode=fast_mode
                )
                for item in batch
            )
        else:
            results = [
                process_single_item_from_maps(
                    item,
                    compute_args,
                    roi_modes=roi_modes,
                    fast_mode=fast_mode
                )
                for item in batch
            ]

        # ---- Flatten + stream to JSONL ----
        with open(output_jsonl, "a") as f:
            for per_item_results in results:
                if not per_item_results:
                    continue
                for r in per_item_results:
                    f.write(json.dumps(r) + "\n")

