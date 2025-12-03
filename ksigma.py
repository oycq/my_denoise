import os
import json
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import sys

# ==========================================
# [MACRO] CONFIGURATION MACROS
# ==========================================
# 图像相关
RAW_HEIGHT = 3000
RAW_WIDTH = 4000
CROP_SIZE = 256           # 新增: 随机裁剪大小

# 归一化参数
INP_SCALE = 12800.0       # 输入归一化除数

# 噪声模型路径
NOISE_MODEL_PATH = 'noise_model.json'

# 数据增强参数 (Augmentation)
AUG_BRIGHTNESS_MIN = 0.5  # [cite: 205] 随机亮度下限
AUG_BRIGHTNESS_MAX = 1.2  # [cite: 205] 随机亮度上限
AUG_GAIN_MIN = 1.0        # 随机 Gain 下限
AUG_GAIN_MAX = 8.0        # 随机 Gain 上限

# ==========================================
# Part 1: k-Sigma Math & Helpers
# ==========================================

def get_noise_parameters(gain):
    if os.path.exists(NOISE_MODEL_PATH):
        with open(NOISE_MODEL_PATH, 'r') as f:
            model = json.load(f)
        a_k = model['k_coefficients']['a']
        b_k = model['k_coefficients']['b']
        k = a_k * gain + b_k
        
        a_sigma2 = model['sigma2_coefficients']['a']
        b_sigma2 = model['sigma2_coefficients']['b']
        c_sigma2 = model['sigma2_coefficients']['c']
        sigma2 = a_sigma2 * (gain ** 2) + b_sigma2 * gain + c_sigma2
    else:
        print("未找到噪模文件")
        sys.exit(0)
    return k, sigma2

def k_sigma_transform(x, k, sigma2, inverse=False):
    k_safe = k + 1e-8
    term_bias = sigma2 / (k_safe ** 2)
    if not inverse:
        return (x / k_safe) + term_bias
    else:
        return k_safe * (x - term_bias)

rng = np.random.default_rng()
def add_noise_transformed(transformed_clean):
    noise_variance = np.maximum(transformed_clean, 0)
    noise_std = np.sqrt(noise_variance)
    standard_noise = rng.standard_normal(transformed_clean.shape, dtype=np.float32)
    noise = standard_noise * noise_std
    return transformed_clean + noise

# ==========================================
# Part 2: Tensor Shape Utils (Pack/Pad)
# ==========================================

def pack_raw_bayer_to_rggb(raw_img):
    # (H, W) -> (4, H/2, W/2)
    H, W = raw_img.shape
    H = (H // 2) * 2
    W = (W // 2) * 2
    raw_img = raw_img[:H, :W]

    R = raw_img[0::2, 0::2]
    Gr = raw_img[0::2, 1::2]
    Gb = raw_img[1::2, 0::2]
    B = raw_img[1::2, 1::2]
    
    return np.stack([R, Gr, Gb, B], axis=0)

def unpack_rggb_to_bayer(rggb):
    # (4, H/2, W/2) -> (H, W)
    C, H_half, W_half = rggb.shape
    H, W = H_half * 2, W_half * 2
    bayer = np.zeros((H, W), dtype=rggb.dtype)
    
    bayer[0::2, 0::2] = rggb[0] # R
    bayer[0::2, 1::2] = rggb[1] # Gr
    bayer[1::2, 0::2] = rggb[2] # Gb
    bayer[1::2, 1::2] = rggb[3] # B
    
    return bayer

# ==========================================
# Part 3: The Dataset
# ==========================================
class RawDenoisingDataset(Dataset):
    def __init__(self, raw_files):
        self.raw_images = []
        for filename in raw_files:
            raw_data = np.fromfile(filename, dtype=np.uint16)
            image = raw_data.reshape((RAW_HEIGHT, RAW_WIDTH))
            image_float = image.astype(np.float32) / 65535.0
            patch_4ch = pack_raw_bayer_to_rggb(image_float)
            self.raw_images.append(patch_4ch)

    def __len__(self):
        return len(self.raw_images) * 1000

    def __getitem__(self, idx):
        # 1. Random img
        file_idx = np.random.randint(0, len(self.raw_images))
        patch_4ch = self.raw_images[file_idx]
        
        # 2. Random Crop to CROP_SIZE
        _, H, W = patch_4ch.shape
        h_start = np.random.randint(0, H - CROP_SIZE)
        w_start = np.random.randint(0, W - CROP_SIZE)
        patch_4ch = patch_4ch[:, h_start:h_start + CROP_SIZE, w_start:w_start + CROP_SIZE]

        # 3. Augmentation (使用宏定义的参数)
        brightness_scale = np.random.uniform(AUG_BRIGHTNESS_MIN, AUG_BRIGHTNESS_MAX)
        patch_4ch = patch_4ch * brightness_scale

        gain = np.random.uniform(AUG_GAIN_MIN, AUG_GAIN_MAX)

        # 4. Transform & Noise
        k, sigma2 = get_noise_parameters(gain)
        
        clean_transformed = k_sigma_transform(patch_4ch, k, sigma2, inverse=False)
        noisy_transformed = add_noise_transformed(clean_transformed)
        
        # 5. Normalize (使用宏定义的 INP_SCALE)
        input_tensor = noisy_transformed / INP_SCALE
        target_tensor = clean_transformed / INP_SCALE

        # 6. Return Data
        params = np.array([k, sigma2, gain], dtype=np.float32)
        return (torch.from_numpy(input_tensor).float(), 
                torch.from_numpy(target_tensor).float(),
                torch.from_numpy(params).float())

# ==========================================
# Part 4: Visualization Utility
# ==========================================

def save_visual_comparison(noisy_tensor, pred_tensor, clean_tensor, params_tensor, save_path):
    """
    可视化工具，自动使用宏定义参数进行逆变换
    """
    k = params_tensor[0].item()
    sigma2 = params_tensor[1].item()
    gain = params_tensor[2].item()

    # 1. Inverse Normalize
    noisy = noisy_tensor.numpy() * INP_SCALE
    pred = pred_tensor.numpy() * INP_SCALE
    clean = clean_tensor.numpy() * INP_SCALE

    # 2. Inverse k-Sigma
    noisy = k_sigma_transform(noisy, k, sigma2, inverse=True)
    pred = k_sigma_transform(pred, k, sigma2, inverse=True)
    clean = k_sigma_transform(clean, k, sigma2, inverse=True)

    # 3. Unpack
    noisy_bayer = unpack_rggb_to_bayer(noisy)
    pred_bayer = unpack_rggb_to_bayer(pred)
    clean_bayer = unpack_rggb_to_bayer(clean)

    # 4. Post Process
    def process_for_display(bayer_img):
        img = np.clip(bayer_img, 0, 1)
        img = np.power(img, 1.0/2.2) # Gamma
        img = (img * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)
        return img_color

    vis_noisy = process_for_display(noisy_bayer)
    vis_pred = process_for_display(pred_bayer)
    vis_clean = process_for_display(clean_bayer)

    # Add Labels
    def add_label(img, text):
        cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return img

    vis_noisy = add_label(vis_noisy, f"Noisy (G:{gain:.1f})")
    vis_pred = add_label(vis_pred, "Denoised")
    vis_clean = add_label(vis_clean, "Clean GT")

    combined = np.hstack((vis_noisy, vis_pred, vis_clean))
    cv2.imwrite(save_path, combined)
