import torch
from torch import autocast
from torch.amp import GradScaler 
from pathlib import Path

class Trainer:
    def __init__(self, train_loader, val_loader, model, loss, optimizer, scheduler, epochs, device, checkpoints, ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.epochs = epochs
        self.device = device

        self.scaler = GradScaler('cuda' if cuda.is_available() else 'cpu')
    
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []

        checkpoints = Path(checkpoints)
        checkpoints.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Epoch {epoch}/{self.epoch} [Train]")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = self.model(x)
                loss = self.loss(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item:.4f}"})

        self.scheduler.step()
        return train_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        pbar = tqdm(self.val_loader, desc="Epoch {epoch}/{self.epoch} [Val]")
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    logtis = self.model(x)
                    loss = self.loss(logits, y)
    
                val_loss += loss.item()
                pbar.set_profix({'Loss': f"{loss.item:.4f}"})
        return val_loss / len(self.val_loader)

    
                    

        
