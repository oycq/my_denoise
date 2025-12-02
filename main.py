import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import net
import ksigma

IF_WANDB = False
PRE_LOAD_PTH = "results_backup/denoise_epoch_99.pth"

def train():
    device = torch.device("cuda")

    # 结果目录
    os.makedirs("results", exist_ok=True)

    # 初始化 WandB
    if IF_WANDB:
        import wandb
        wandb.init(project="denoising-project")

    # Dataset (参数都在 ksigma 宏里定义了)
    dataset = ksigma.RawDenoisingDataset(glob.glob("*.raw"))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=47)

    # 模型
    model = net.Network().to(device)
    if PRE_LOAD_PTH:
        checkpoint = torch.load(PRE_LOAD_PTH, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    num_epochs = 100

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets, params) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = torch.abs(outputs - targets).mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Step [{batch_idx}] RMS Loss: {loss.item():.6f}   {torch.abs(inputs - targets).mean():.6f}")

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

        # 记录到 WandB
        if IF_WANDB:
            wandb.log({"epoch": epoch, "avg_loss": avg_loss})

if __name__ == "__main__":
    train()