# dendro_cli.py
import os, math, json, argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import scipy.stats as stats

# optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from diffusers.models import AutoencoderKL


# ============== Utils (viz only; core math unchanged) ==============
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.min() >= -1.01 and x.max() <= 1.01:
        x = (x + 1.0) / 2.0
    else:
        lo, hi = x.min(), x.max()
        x = (x - lo) / (hi - lo + 1e-8)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def save_tensor_image(t: torch.Tensor, path: str):
    arr = t.detach().cpu().numpy()
    if arr.ndim == 2:
        Image.fromarray(to_uint8(arr)).save(path); return
    if arr.ndim == 3:
        C, H, W = arr.shape
        if C == 1:
            Image.fromarray(to_uint8(arr[0])).save(path); return
        if C in (3, 4):
            Image.fromarray(to_uint8(np.transpose(arr, (1, 2, 0)))).save(path); return
    arr2 = np.squeeze(arr)
    if arr2.ndim == 2:
        Image.fromarray(to_uint8(arr2)).save(path)
    else:
        raise ValueError(f"unexpected shape for image save: {arr.shape}")

def save_fft_mag_image(t: torch.Tensor, path: str):
    # visualization-only: FFT done in float32
    if isinstance(t, torch.Tensor): t = t.detach().cpu().numpy()
    mag = np.log(np.abs(t).astype(np.float32) + 1e-6)
    Image.fromarray(to_uint8(mag)).save(path)

def fft2_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(x.float())

def ifft2_real_like(xc: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(xc).real.to(like.dtype)

def psnr(a_img: Image.Image, b_img: Image.Image) -> float:
    a = np.asarray(a_img).astype(np.float32)
    b = np.asarray(b_img).astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12: return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))

def ssim_safe(a_img: Image.Image, b_img: Image.Image):
    if not HAVE_SKIMAGE: return None
    a = np.asarray(a_img); b = np.asarray(b_img)
    try:
        return float(ssim(a, b, data_range=255, channel_axis=2))
    except TypeError:
        return float(ssim(a, b, data_range=255, multichannel=True))


# ============== Watermark helpers (same core mask logic) ==============
def circle_mask(size=128, r=16):
    y, x = np.ogrid[:size, :size]
    c = size // 2
    return ((x - c) ** 2 + (y - c) ** 2) <= (r ** 2)

def get_pattern(pipe, shape, size, w_seed):
    # build key via FFT of random latents (float32 FFT for stability)
    g = torch.Generator(device=pipe.device); g.manual_seed(w_seed)
    gt_init = pipe.prepare_latents(
        1, pipe.unet.config.in_channels, size, size,
        pipe.unet.dtype, pipe.device, g
    )
    gt_fft = torch.fft.fftshift(fft2_f32(gt_init), dim=(-1, -2))
    gt_fft_tmp = gt_fft.clone().detach()
    for i in range(shape[-1] // 2, 0, -1):
        tmp_mask = torch.tensor(circle_mask(gt_init.shape[-1], r=i),
                                device=pipe.device, dtype=torch.bool)
        for j in range(gt_fft.shape[1]):
            gt_fft[:, j, tmp_mask] = gt_fft_tmp[0, j, 0, i].item()
    return gt_fft  # complex64

def build_wm_key_mask(pipe, size, w_radius, w_channel, w_seed, wm_cell=8):
    lh = lw = size // wm_cell
    shape = (1, 4, lh, lw)
    np_mask = circle_mask(lw, r=w_radius)
    torch_mask = torch.tensor(np_mask, device=pipe.device, dtype=torch.bool)
    w_mask = torch.zeros(shape, dtype=torch.bool, device=pipe.device)
    w_mask[:, w_channel] = torch_mask
    w_key = get_pattern(pipe, shape, size, w_seed).to(pipe.device)   # complex64
    return w_key, w_mask, shape, torch_mask


# ============== Detection (core logic preserved) ==============
@torch.inference_mode()
def detect_score(pipe, image_pil: Image.Image, w_key, w_mask, steps_invert: int, size: int) -> dict:
    curr = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(curr.config)

    img = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])(image_pil)
    img = (2.0 * img - 1.0).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.13025

    inv = pipe(prompt="", latents=image_latents, guidance_scale=1,
               num_inference_steps=steps_invert, output_type="latent")
    inverted_latents = inv.images  # (1,4,H/8,W/8)

    # viz FFT (float32)
    inv_fft_f32 = torch.fft.fftshift(fft2_f32(inverted_latents), dim=(-1, -2))

    # core stats (fft at native dtype)
    inv_fft_core = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
    inv_masked = inv_fft_core[w_mask].flatten()
    target = w_key.to(inv_masked.dtype)[w_mask].flatten()  # <-- dtype aligned

    inv_cat = torch.cat([inv_masked.real, inv_masked.imag])
    tgt_cat = torch.cat([target.real, target.imag])

    sigma = inv_cat.std()
    sigma = sigma + 1e-12
    lamda = (tgt_cat ** 2 / (sigma ** 2)).sum().item()
    x = (((inv_cat - tgt_cat) / sigma) ** 2).sum().item()
    p_value = stats.ncx2.cdf(x=x, df=len(tgt_cat), nc=lamda)
    score = 1.0 if p_value == 0 else max(0.0, 1 - 1 / math.log(5 / p_value, 10))

    pipe.scheduler = curr
    return {"p_value": float(p_value), "score": float(score), "inv_fft_f32": inv_fft_f32}


# ============== Main experiment ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--negative", type=str, default="monochrome")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--steps-invert", type=int, default=50)
    ap.add_argument("--w-seed", type=int, default=7433)
    ap.add_argument("--w-radius", type=int, default=16)   # 1024→16, 768→12
    ap.add_argument("--w-channel", type=int, default=0)
    ap.add_argument("--wm-cell", type=int, default=8)     # SDXL latent downsample
    ap.add_argument("--outdir", type=str, default="runs/exp1")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # pipeline
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained("mhdang/dpo-sdxl-text2image-v1",
                                                subfolder="unet", torch_dtype=dtype)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=unet, vae=vae, torch_dtype=dtype
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if not args.no_compile and device == "cuda":
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(f"[warn] torch.compile skipped: {e}")

    # watermark key & mask
    w_key, w_mask, shape, torch_mask = build_wm_key_mask(
        pipe, args.size, args.w_radius, args.w_channel, args.w_seed, wm_cell=args.wm_cell
    )
    save_tensor_image(torch_mask.float().cpu(), os.path.join(args.outdir, "wm_mask.png"))

    # base latents for fair comparison (same seed path)
    lat_base = pipe.prepare_latents(
        1, pipe.unet.config.in_channels, args.size, args.size,
        pipe.unet.dtype, pipe.device, None
    )
    save_tensor_image(lat_base[0, 0], os.path.join(args.outdir, "latent_init.png"))

    # visualize FFT before (float32)
    lat_fft_before_f32 = torch.fft.fftshift(fft2_f32(lat_base), dim=(-1, -2))
    save_fft_mag_image(lat_fft_before_f32[0, 0], os.path.join(args.outdir, "fft_before.png"))

    # inject watermark (core)
    latent_h, latent_w = lat_base.shape[-2:]
    power2_ok = ((latent_h & (latent_h - 1)) == 0) and ((latent_w & (latent_w - 1)) == 0)
    use_safe_fft = (dtype == torch.float16 and device == "cuda" and not power2_ok)

    lat_wm = lat_base.clone()
    if use_safe_fft:
        fft = torch.fft.fftshift(fft2_f32(lat_wm), dim=(-1, -2))                    # complex64
        fft_dtype = fft.dtype
        fft[w_mask] = w_key.to(fft_dtype)[w_mask].clone()                            # dtype aligned
        lat_wm = ifft2_real_like(torch.fft.ifftshift(fft, dim=(-1, -2)), lat_wm)
    else:
        fft = torch.fft.fftshift(torch.fft.fft2(lat_wm), dim=(-1, -2))              # complex16
        fft_dtype = fft.dtype
        fft[w_mask] = w_key.to(fft_dtype)[w_mask].clone()                            # dtype aligned
        lat_wm = torch.fft.ifft2(torch.fft.ifftshift(fft, dim=(-1, -2))).real
        lat_wm = torch.nan_to_num(lat_wm, nan=0.0, posinf=4.0, neginf=-4.0)

    # visualize FFT after (float32)
    lat_fft_after_f32 = torch.fft.fftshift(fft2_f32(lat_wm), dim=(-1, -2))
    save_fft_mag_image(lat_fft_after_f32[0, 0], os.path.join(args.outdir, "fft_after.png"))

    # difference (viz only)
    diff_f32 = torch.abs(lat_fft_after_f32 - lat_fft_before_f32).mean(dim=1, keepdim=True)
    save_fft_mag_image(diff_f32[0, 0], os.path.join(args.outdir, "fft_delta.png"))

    @torch.inference_mode()
    def gen_from_lat(lat):
        out = pipe(prompt=args.prompt, negative_prompt=args.negative,
                   num_inference_steps=args.steps, latents=lat)
        return out.images[0]

    img_no = gen_from_lat(lat_base)
    img_wm = gen_from_lat(lat_wm)

    no_path = os.path.join(args.outdir, "no_wm.png")
    wm_path = os.path.join(args.outdir, "with_wm.png")
    img_no.save(no_path); img_wm.save(wm_path)

    # metrics
    psnr_v = psnr(img_no, img_wm)
    ssim_v = ssim_safe(img_no, img_wm)

    # detection
    det_no = detect_score(pipe, img_no, w_key, w_mask, args.steps_invert, args.size)
    det_wm = detect_score(pipe, img_wm, w_key, w_mask, args.steps_invert, args.size)

    # save inverted FFT viz for WM case
    save_fft_mag_image(det_wm["inv_fft_f32"][0, 0], os.path.join(args.outdir, "fft_inverted.png"))

    metrics = {
        "prompt": args.prompt,
        "size": args.size,
        "steps_gen": args.steps,
        "steps_invert": args.steps_invert,
        "w_seed": args.w_seed,
        "w_radius": args.w_radius,
        "w_channel": args.w_channel,
        "psnr": float(psnr_v),
        "ssim": (None if ssim_v is None else float(ssim_v)),
        "detect_no_watermark": {"p_value": det_no["p_value"], "score": det_no["score"]},
        "detect_with_watermark": {"p_value": det_wm["p_value"], "score": det_wm["score"]},
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # safe markdown writer (no f-string multiline pitfalls)
    md_lines = []
    md_lines.append("# 🧬 Dendrokronos CLI Report\n")
    md_lines.append(f"**Prompt:** `{args.prompt}`  \n")
    md_lines.append(f"**Resolution:** {args.size}×{args.size}  \n")
    md_lines.append(f"**Steps (gen / invert):** {args.steps} / {args.steps_invert}  \n")
    md_lines.append(f"**Watermark:** seed={args.w_seed}, radius={args.w_radius}, channel={args.w_channel}\n")
    md_lines.append("\n---\n\n## 1) Image Comparison\n")
    md_lines.append("| No Watermark | With Watermark |\n|---|---|\n")
    md_lines.append("| ![NoWM](no_wm.png) | ![WithWM](with_wm.png) |\n\n")
    md_lines.append(f"**PSNR:** {psnr_v:.2f} dB  \n")
    if ssim_v is not None:
        md_lines.append(f"**SSIM:** {ssim_v:.4f}  \n")
    else:
        md_lines.append("**SSIM:** (skimage not installed; run `pip install scikit-image`)  \n")
    md_lines.append("> PSNR > 35 dB ≈ visual difference nearly invisible.\n\n")
    md_lines.append("---\n\n## 2) Latent Frequency (log magnitude)\n")
    md_lines.append("- **Mask:** ![mask](wm_mask.png)\n\n")
    md_lines.append("| Before | After | Δ (mean |before–after| over channels) |\n|---|---|---|\n")
    md_lines.append("| ![before](fft_before.png) | ![after](fft_after.png) | ![delta](fft_delta.png) |\n\n")
    md_lines.append("---\n\n## 3) Detection (DDIM inversion)\n")
    md_lines.append(f"- **No watermark:** p={det_no['p_value']:.3e}, score={det_no['score']:.3f}  \n")
    md_lines.append(f"- **With watermark:** p={det_wm['p_value']:.3e}, score={det_wm['score']:.3f}  \n\n")
    md_lines.append("Interpretation:\n- score ≥ 0.7 → strong positive\n- 0.5–0.7 → borderline\n- < 0.5 → likely unmarked\n\n")
    md_lines.append("> Inverted frequency (watermarked case):\n![inv](fft_inverted.png)\n\n")
    md_lines.append("*Generated by `dendro_cli.py` (visualizations in float32 FFT; core watermark/detect math unchanged).*")

    with open(os.path.join(args.outdir, "report.md"), "w") as f:
        f.write("".join(md_lines))

    print(json.dumps(metrics, indent=2))
    print(f"\n✅ Outputs in: {args.outdir}\n"
          "   - report.md, metrics.json\n"
          "   - no_wm.png / with_wm.png\n"
          "   - wm_mask.png, latent_init.png, fft_before.png, fft_after.png, fft_delta.png, fft_inverted.png")

if __name__ == "__main__":
    main()
