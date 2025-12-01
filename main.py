import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import net
import ksigma
import sys

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 检查数据
    raw_files = glob.glob("*.raw")
    if not raw_files:
        print("No .raw files found")
        sys.exit(0)

    # 结果目录
    os.makedirs("results", exist_ok=True)

    # Dataset (参数都在 ksigma 宏里定义了)
    dataset = ksigma.RawDenoisingDataset(raw_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # 模型
    model = net.Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse_criterion = nn.MSELoss()

    num_epochs = 100

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets, params) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # [cite_start]Loss [cite: 208]
            loss = torch.sqrt(mse_criterion(outputs, targets))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Step [{batch_idx}] RMS Loss: {loss.item():.6f}")

        # 存图
        n_t = inputs[0].detach().cpu()
        p_t = outputs[0].detach().cpu()
        c_t = targets[0].detach().cpu()
        par_t = params[0].detach().cpu()
        save_name = f"results/epoch_{epoch}.jpg"
        ksigma.save_visual_comparison(n_t, p_t, c_t, par_t, save_name)
        # 打印损失函数
        avg_loss = epoch_loss / len(dataloader)
        print(f"=== Epoch {epoch} Done. Avg Loss: {avg_loss:.6f} ===")
        # 保存模型
        torch.save(model.state_dict(), f"results/denoise_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()