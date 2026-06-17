import os
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import autocast
from torch.amp import GradScaler 
from tqdm import tqdm  

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.dataset import FastAllenCell2DDataset
from src.models.unet import UNet
from src.losses.losses import get_loss_fn

def save_training_curves(train_losses, val_losses, save_dir='./data'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='tab:blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', color='tab:red', linewidth=2)

    plt.title('U-Net Cell Segmentation - Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)

    plt.ylabel('Loss', fontsize=12) 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)

    save_path = os.path.join(save_dir, 'training_curves.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):

    BATCH_SIZE = cfg.train.batch_size
    EPOCHS = cfg.train.epochs
    LR = cfg.train.lr
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('checkpoints', exist_ok=True)


    train_dataset = FastAllenCell2DDataset(cfg.data.train_files, input_ch=0, target_ch=4)
    val_dataset = FastAllenCell2DDataset(cfg.data.val_files, input_ch=0, target_ch=4)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False, 
    )


    model = UNet(
        in_channels=cfg.model.in_channels, 
        out_channels=cfg.model.out_channels, 
        init_features=cfg.model.init_features,
        depth=cfg.model.depth
    ).to(DEVICE)


    criterion = get_loss_fn(cfg.loss)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_loss = 0.0

        pbar_train = tqdm(train_dataloader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')

        for x, y in pbar_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                preds = model(x)
                loss = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_dataloader)
        train_loss_history.append(avg_train_loss)
        

        scheduler.step()


        model.eval()
        val_loss = 0.0 

        pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")

        with torch.no_grad():
            for x, y in pbar_val:
                x, y = x.to(DEVICE), y.to(DEVICE)


                with autocast(device_type='cuda', dtype=torch.float16):
                    preds = model(x)
                    loss = criterion(preds, y)

                val_loss += loss.item()
                pbar_val.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_dataloader)
        val_loss_history.append(avg_val_loss)

        print(f'Epoch {epoch} Summary | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')


        save_training_curves(train_loss_history, val_loss_history, save_dir='./data')

        if avg_val_loss < best_val_loss:
            print(f'Val Loss Improved: {best_val_loss:.4f} -> {avg_val_loss:.4f}. Saving model...')
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_unet.pth')
        else:
            print()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()