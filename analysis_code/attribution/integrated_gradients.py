"""Integrated Gradients with mean-image and blurred-image reference baselines."""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(kernel_size, sigma, device):
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur(img, sigma):
    """
    img: [1, C, H, W]
    """
    kernel_size = int(2 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    device = img.device
    C = img.shape[1]

    kernel = gaussian_kernel(kernel_size, sigma, device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(C, 1, 1, 1)  # depthwise

    return F.conv2d(
        img,
        kernel,
        padding=kernel_size // 2,
        groups=C
    )

def get_baseline(img_stream, mode="blur", sigma=10, mean_img=None):
    if mode == "black":
        return torch.zeros_like(img_stream)
    elif mode == "blur":
        return gaussian_blur(img_stream, sigma=sigma)
    elif mode == "mean":
        assert mean_img is not None
        return mean_img
    elif mode == "heavy_blur":
        return gaussian_blur(img_stream, sigma=48)
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")

def integrated_gradients_single_stream(
    model,
    img_stream,
    other_stream,
    stream_id,
    steps=64,
    baseline_mode="blur",
    baseline_sigma=10,
    mean_img=None
):
    metric = None
    
    baseline = get_baseline(
        img_stream,
        mode=baseline_mode,
        sigma=baseline_sigma,
        mean_img=mean_img
    )

    total_grad = torch.zeros_like(img_stream)

    for i in range(1, steps + 1):
        alpha = float(i) / steps
        interp_img = baseline + alpha * (img_stream - baseline)
        interp_img.requires_grad_(True)

        if stream_id == 0:
            imgs = torch.stack([interp_img, other_stream], dim=1)
        else:
            imgs = torch.stack([other_stream, interp_img], dim=1)

        model.zero_grad()
        logits, _ = model([imgs, metric] if metric is not None else imgs)
        score = logits[:, 0].sum()
        score.backward()

        total_grad += interp_img.grad.detach()

    avg_grad = total_grad / steps
    ig = (img_stream - baseline) * avg_grad

    heatmap = ig.abs().mean(dim=1)  # [1, H, W]
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)

    return heatmap

def normalize(x):
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

def to_gray(img):
    # img: [3, H, W] or [H, W]
    if img.ndim == 3:
        return normalize(img.mean(axis=0))
    return normalize(img)

def enhance_heatmap(heatmap, gamma=0.7, clip_percentile=99):
    """
    Makes IG visually clearer without changing topology
    """
    h = heatmap.copy()
    p_low, p_high = np.percentile(h, [5, 99])
    h = np.clip(h, p_low, p_high)
    h = normalize(h)
    h = h ** gamma   # gamma < 1 enhances thin vessels
    return h

def plot_dual_eye_ig_multibaseline(
    img_left,
    img_right,
    ig_left_dict,
    ig_right_dict,
    gamma=0.6,
    clip_percentile=99,
    figsize=(14, 6),
    title=None
):
    """
    Column order:
    Original | IG (mean) | IG (heavy blur σ=48) | IG (blur σ=10)
    """

    # ---- helpers ----
    def prep(hm):
        return enhance_heatmap(hm, gamma=gamma, clip_percentile=clip_percentile)

    img_l = to_gray(img_left)
    img_r = to_gray(img_right)

    ig_l_mean  = prep(ig_left_dict["mean"])
    ig_l_hblur = prep(ig_left_dict["heavy_blur"])
    ig_l_blur  = prep(ig_left_dict["blur"])

    ig_r_mean  = prep(ig_right_dict["mean"])
    ig_r_hblur = prep(ig_right_dict["heavy_blur"])
    ig_r_blur  = prep(ig_right_dict["blur"])

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # ===== LEFT EYE =====
    axes[0, 0].imshow(img_l, cmap="gray")
    axes[0, 0].set_title("Left eye\nOriginal")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ig_l_mean, cmap="hot")
    axes[0, 1].set_title("IG (mean baseline)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(ig_l_hblur, cmap="hot")
    axes[0, 2].set_title("IG (heavy blur σ=48)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(ig_l_blur, cmap="hot")
    axes[0, 3].set_title("IG (blur σ=10)")
    axes[0, 3].axis("off")

    # ===== RIGHT EYE =====
    axes[1, 0].imshow(img_r, cmap="gray")
    axes[1, 0].set_title("Right eye\nOriginal")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(ig_r_mean, cmap="hot")
    axes[1, 1].set_title("IG (mean baseline)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(ig_r_hblur, cmap="hot")
    axes[1, 2].set_title("IG (heavy blur σ=48)")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(ig_r_blur, cmap="hot")
    axes[1, 3].set_title("IG (blur σ=10)")
    axes[1, 3].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()
