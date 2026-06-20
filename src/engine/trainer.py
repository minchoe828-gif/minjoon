import torch
from torch import autocast
from torch.amp import GradScaler 
from pathlib import Path
from loguru import logger
from tqdm import tqdm             
import matplotlib.pyplot as plt   

class Trainer:
    def __init__(self, train_loader, val_loader, model, loss, optimizer, scheduler, epochs, device, checkpoints):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.epochs = epochs
        self.device = device

        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []

        self.checkpoints_dir = Path(checkpoints)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = self.model(x)
                loss = self.loss(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        self.scheduler.step()
        return train_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
        
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model(x)
                    loss = self.loss(logits, y)
    
                val_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
        return val_loss / len(self.val_loader)

    def save_curves(self):
        epochs_range = range(1, len(self.train_loss_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.train_loss_history, label='Train Loss', color='tab:blue', linewidth=2)
        plt.plot(epochs_range, self.val_loss_history, label='Val Loss', color='tab:red', linewidth=2)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # 임시: 추후 커스텀 이름 적용
        plt.savefig(self.checkpoints_dir / 'training_curves.png', dpi=300, bbox_inches='tight') 
        plt.close()

    def run(self):
        for epoch in range(1, self.epochs + 1):
            avg_train_loss = self.train_epoch(epoch)
            avg_val_loss = self.validate_epoch(epoch)

            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)

            logger.info(f"Epoch {epoch} Summary | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

            if avg_val_loss < self.best_val_loss:
                logger.info(f"Val Loss Improved: {self.best_val_loss:.4f} -> {avg_val_loss:.4f}")
                self.best_val_loss = avg_val_loss
                
                # 임시: 추후 커스텀 이름 적용
                save_path = self.checkpoints_dir / 'best_model.pth'
                torch.save(self.model.state_dict(), str(save_path)) 
                
        self.save_curves()
        
        

    
                    

        
