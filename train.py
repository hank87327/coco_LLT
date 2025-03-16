import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.dataset import get_dataloaders
from models.classifier import create_model
from utils.metrics import evaluate_model, print_metrics, plot_confusion_matrix
from utils.visualization import visualize_batch, plot_training_history
from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, 
    LR_SCHEDULER, CHECKPOINT_DIR, LOG_DIR, STEP_SIZE, GAMMA,
    RANDOM_SEED
)

def set_seed(seed=RANDOM_SEED):
    """設置隨機種子以確保結果可重複"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model():
    """訓練 COCO 分類模型"""
    # 設置隨機種子
    set_seed()
    
    # 載入數據
    print("載入數據...")
    dataloaders = get_dataloaders()
    
    # 創建模型
    print(f"創建模型 (使用設備: {DEVICE})...")
    model = create_model()
    model = model.to(DEVICE)
    
    # 定義損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 學習率調整器
    if LR_SCHEDULER == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    else:
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # 訓練歷史記錄
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0,
        'best_acc': 0.0
    }
    
    # 創建日誌目錄
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 創建檢查點目錄
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 可視化一批訓練數據
    train_inputs, train_labels = next(iter(dataloaders['train']))
    visualize_batch(
        train_inputs, train_labels, 
        save_path=os.path.join(LOG_DIR, 'train_batch.png')
    )
    
    # 開始訓練
    print(f"開始訓練 ({NUM_EPOCHS} 個 epochs)...")
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        # 使用 tqdm 顯示進度條
        train_loader = tqdm(dataloaders['train'], desc="Training")
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 反向傳播與優化
            loss.backward()
            optimizer.step()
            
            # 統計
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
            # 更新進度條
            train_loader.set_postfix({
                'loss': loss.item(),
                'acc': torch.sum(preds == labels.data).item() / inputs.size(0)
            })
        
        # 計算訓練指標
        epoch_train_loss = train_loss / len(dataloaders['train'].dataset)
        epoch_train_acc = train_corrects.double() / len(dataloaders['train'].dataset)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        
        # 驗證階段
        print("驗證中...")
        val_metrics = evaluate_model(model, dataloaders['val'], DEVICE, criterion)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印進度
        print(f"訓練損失: {epoch_train_loss:.4f}, 訓練準確率: {epoch_train_acc:.4f}")
        print(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")
        
        # 檢查每個類別的準確率
        print("每個類別的準確率:")
        for category, acc in val_metrics['class_accuracy'].items():
            print(f"  {category}: {acc:.4f}")
        
        # 學習率調整
        current_lr = optimizer.param_groups[0]['lr']
        print(f"當前學習率: {current_lr:.6f}")
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_epoch'] = epoch
            history['best_acc'] = val_acc
            
            # 保存最佳模型
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"保存最佳模型到 {best_model_path} (準確率: {val_acc:.4f})")
            
            # 繪製混淆矩陣
            cm_path = os.path.join(LOG_DIR, f'confusion_matrix_epoch{epoch+1}.png')
            plot_confusion_matrix(val_metrics, save_path=cm_path)
        
        # 每個 epoch 保存檢查點
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"保存檢查點到 {checkpoint_path}")
        
        print("-" * 50)
    
    # 訓練完成
    time_elapsed = time.time() - start_time
    print(f"訓練完成，耗時 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"最佳驗證準確率: {best_val_acc:.4f} at epoch {history['best_epoch']+1}")
    
    # 繪製訓練歷史
    history_path = os.path.join(LOG_DIR, 'training_history.png')
    plot_training_history(history, save_path=history_path)
    
    # 保存訓練歷史到 JSON
    history_json = {k: [float(v) if isinstance(v, (np.float32, np.float64, np.int64)) else v for v in vals] 
                    if isinstance(vals, list) else float(vals) if isinstance(vals, (np.float32, np.float64, np.int64)) else vals 
                    for k, vals in history.items()}
    
    with open(os.path.join(LOG_DIR, 'training_history.json'), 'w') as f:
        json.dump(history_json, f, indent=2)
    
    return model, history

if __name__ == "__main__":
    train_model()
