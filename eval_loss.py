import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np

# 引入你的工程模块
import net
import ksigma

# --- 配置参数 ---
MODEL_PATH = "results/denoise_epoch_99.pth"  # 请修改为实际路径
DATA_PATH = "train_data/*.raw"               # 验证集路径
OUTPUT_FILE = "output.txt"
BATCH_SIZE = 16
GAIN_INDEX = 2        # params中的增益索引
PIXEL_BINS = 1024     # 亮度分级数量
MIN_PIXEL_COUNT = 100 # 统计阈值
EPS = 1e-4            # 防止除零 (GT为0时)

def evaluate_simple_stats():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 1. 准备统计容器
    # ==========================================
    
    # --- A. Gain Level (整图统计) ---
    gain_ranges = [(i, i + 1) for i in range(1, 8)]
    # 存储字典，key是区间，value是列表，列表里存元组 (l1, rel_err)
    gain_buckets_stats = defaultdict(list)

    # --- B. Pixel Level (像素级统计 - GPU累加器) ---
    # 只需要维护3个累加器
    # 1. L1 误差和: sum(|x-y|)
    pixel_l1_sum = torch.zeros(PIXEL_BINS, device=device, dtype=torch.float64)
    # 2. 相对误差和: sum(|x-y| / (y + eps))
    pixel_rel_sum = torch.zeros(PIXEL_BINS, device=device, dtype=torch.float64)
    # 3. 计数器
    pixel_count = torch.zeros(PIXEL_BINS, device=device, dtype=torch.float64)

    # ==========================================
    # 2. 加载模型
    # ==========================================
    print(f"Loading model from {MODEL_PATH}...")
    model = net.Network().to(device)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
    else:
        print(f"Error: Model path {MODEL_PATH} not found.")
        return
    model.eval()

    # ==========================================
    # 3. 数据加载与推理
    # ==========================================
    file_list = glob.glob(DATA_PATH)
    if not file_list:
        print("Error: No data found.")
        return
    print(f"Found {len(file_list)} images. Starting evaluation...")
    
    dataset = ksigma.RawDenoisingDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets, params) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            params = params.to(device)

            outputs = model(inputs)
            
            # 基础差异
            abs_diff = torch.abs(outputs - targets)

            # ===============================
            #  统计 Part 1: 按 Gain (Image Metrics)
            # ===============================
            # L1 per image: [B]
            img_l1 = torch.mean(abs_diff, dim=[1, 2, 3])
            
            # Relative Error per image: mean(|diff| / (gt + eps)) * 100
            # 计算整张图的平均误差百分比
            img_rel_err = torch.mean(abs_diff / (targets + EPS), dim=[1, 2, 3]) * 100.0

            current_gains = params[:, GAIN_INDEX]

            for i in range(len(inputs)):
                g = current_gains[i].item()
                # 打包数据 (L1, Err%)
                stats = (img_l1[i].item(), img_rel_err[i].item())
                
                for r in gain_ranges:
                    if r[0] <= g < r[1]:
                        gain_buckets_stats[r].append(stats)
                        break

            # ===============================
            #  统计 Part 2: 按 Pixel (Pixel Metrics)
            # ===============================
            flat_targets = targets.view(-1)      # [N]
            flat_abs_diff = abs_diff.view(-1)    # [N]
            
            # 计算每个像素的相对误差 (0.05 代表 5%)
            flat_rel_err = flat_abs_diff / (flat_targets + EPS)

            # 计算 bin 索引 (0 ~ 1023)
            bin_indices = (flat_targets * PIXEL_BINS).long().clamp(0, PIXEL_BINS - 1)
            
            # --- GPU 并行累加 ---
            # 1. 累加 L1
            pixel_l1_sum.index_add_(0, bin_indices, flat_abs_diff.double())
            # 2. 累加 Relative Error
            pixel_rel_sum.index_add_(0, bin_indices, flat_rel_err.double())
            # 3. 计数
            pixel_count.index_add_(0, bin_indices, torch.ones_like(flat_abs_diff, dtype=torch.float64))

            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}...")

    # ==========================================
    # 4. 结果计算与输出
    # ==========================================
    print("Calculating final statistics...")
    
    results_str = []
    results_str.append("======================================================================")
    results_str.append(f"Model: {MODEL_PATH}")
    results_str.append("Metrics:")
    results_str.append("  - Avg L1:   Mean Absolute Error")
    results_str.append("  - Avg Err%: Mean Relative Error Percentage (|Out-GT|/GT * 100)")
    results_str.append("======================================================================\n")

    # --- Part A: Gain 结果 ---
    results_str.append(">>> Section 1: Statistics by Gain (ISO) <<<")
    # 格式化表头
    header_gain = f"{'Gain Range':<12} | {'Count':<6} | {'Avg L1':<12} | {'Avg Err%':<12}"
    results_str.append(header_gain)
    results_str.append("-" * len(header_gain))
    
    for r in gain_ranges:
        stats_list = gain_buckets_stats[r] # list of (l1, rel)
        count = len(stats_list)
        if count > 0:
            l1s, rels = zip(*stats_list)
            avg_l1 = sum(l1s) / count
            avg_rel = sum(rels) / count
            
            line = f"{r[0]}-{r[1]:<10} | {count:<6} | {avg_l1:.6f}     | {avg_rel:.4f}%"
            results_str.append(line)
        else:
            results_str.append(f"{r[0]}-{r[1]:<10} | {0:<6} | N/A          | N/A")
    results_str.append("\n" + "="*70 + "\n")

    # --- Part B: Pixel Intensity 结果 ---
    results_str.append(">>> Section 2: Statistics by Pixel Intensity (0-1023) <<<")
    # 格式化表头
    header_pixel = f"{'Pixel Range':<20} | {'Count':<10} | {'Avg L1':<12} | {'Avg Err%':<12}"
    results_str.append(header_pixel)
    results_str.append("-" * len(header_pixel))

    # 转回 CPU 处理
    p_l1 = pixel_l1_sum.cpu().numpy()
    p_rel = pixel_rel_sum.cpu().numpy()
    p_cnt = pixel_count.cpu().numpy()

    for i in range(PIXEL_BINS):
        count = p_cnt[i]
        
        range_label = f"{i}/1024 - {i+1}/1024"
        
        if count >= MIN_PIXEL_COUNT:
            # 1. 平均 L1
            avg_l1 = p_l1[i] / count
            
            # 2. 平均误差百分比 (需要 * 100)
            avg_rel_percent = (p_rel[i] / count) * 100.0
            
            line = f"{range_label:<20} | {int(count):<10} | {avg_l1:.6f}     | {avg_rel_percent:.4f}%"
            results_str.append(line)
        else:
            # 数量不足
            line = f"{range_label:<20} | {int(count):<10} | {'-'*12} | {'-'*12}"
            results_str.append(line)

    # --- 5. 写入文件 ---
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results_str))
    
    print(f"\nAll Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate_simple_stats()