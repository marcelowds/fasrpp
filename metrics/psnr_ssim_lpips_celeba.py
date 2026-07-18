import os
import glob
import numpy as np
from PIL import Image
import torch
import lpips
import core.metrics as Metrics
from skimage.metrics import structural_similarity as ssim_

# ==============================
# CONFIGURAÇÕES
# ==============================
HR_DIR = "../sample_imgs/probe_HR"

ALGORITHMS = {
    "FASR++": "../exps/sr_results/*sr_mean.png",
    }

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def load_image_np(path):
    """Carrega imagem como numpy [0,255]."""
    return np.array(Image.open(path).convert("RGB"))

def load_image_tensor(path):
    """Carrega imagem como tensor [-1,1] para LPIPS."""
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return img.to(DEVICE)

def match_images(hr_dir, pattern):
    """Cria pares HR/SR baseando-se no índice numérico."""
    sr_files = sorted(glob.glob(pattern))
    hr_files = []
    for sr_file in sr_files:
        base = os.path.basename(sr_file)
        num = ''.join([c for c in base if c.isdigit()])[:6]  # captura o índice "000001"
        hr_file = os.path.join(hr_dir, f"{num}.jpg")
        if os.path.exists(hr_file):
            hr_files.append(hr_file)
        else:
            print(f"[Aviso] HR ausente para {sr_file}")
    return hr_files, sr_files

# ==============================
# FUNÇÃO PRINCIPAL
# ==============================
def calculate_metrics(hr_list, sr_list, lpips_fn):
    psnrs, ssims, lpipss = [], [], []

    for hr_path, sr_path in zip(hr_list, sr_list):
        hr_img = load_image_np(hr_path)
        sr_img = load_image_np(sr_path)

        # Redimensiona se necessário
        if hr_img.shape != sr_img.shape:
            sr_img = np.array(Image.fromarray(sr_img).resize((hr_img.shape[1], hr_img.shape[0])))

        # ---- PSNR e SSIM ----
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        #ssim = Metrics.calculate_ssim(sr_img, hr_img)
        ssim = ssim_(sr_img, hr_img, channel_axis=-1, data_range=255)

        # ---- LPIPS ----
        hr_tensor = load_image_tensor(hr_path)
        sr_tensor = load_image_tensor(sr_path)
        dist = lpips_fn(sr_tensor, hr_tensor)
        lpips_val = dist.item()

        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lpips_val)

    return (
        np.mean(psnrs), np.std(psnrs),
        np.mean(ssims), np.std(ssims),
        np.mean(lpipss), np.std(lpipss)
    )


# ==============================
# EXECUTION
# ==============================
if __name__ == "__main__":
    print("=== PSNR, SSIM and LPIPS Evaluation ===")
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

    results = []

    for name, pattern in ALGORITHMS.items():
        print(f"\n>> Evaluating {name}")
        hr_list, sr_list = match_images(HR_DIR, pattern)
        if not hr_list:
            print(f"No images found for {name}.")
            continue
        mean_psnr, std_psnr, mean_ssim, std_ssim, mean_lpips, std_lpips = calculate_metrics(hr_list, sr_list, lpips_fn)
        results.append((name, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_lpips, std_lpips))

    print("\n=== FINAL RESULTS ===")
    print(f"{'Method':<20} {'Mean PSNR':>18} {'Mean SSIM':>20} {'Mean LPIPS':>20}")
    print("-" * 70)
    for (name, mpsnr, spsnr, mssim, sssim, mlp, slp) in results:
        print(f"{name:<20} PSNR: {mpsnr:6.4f} ± {spsnr:6.4f},  "
              f"SSIM: {mssim:6.4f} ± {sssim:6.4f},  "
              f"LPIPS: {mlp:6.4f} ± {slp:6.4f}")
