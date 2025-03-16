import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
import os
import sys
sys.path.append('..')
from config import SELECTED_CATEGORIES
import matplotlib
matplotlib.use('Agg')  # 無頭模式，適合在伺服器環境中使用

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反標準化圖像張量"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_batch(inputs, labels, preds=None, num_images=8, save_path=None):
    """可視化一個批次的圖像"""
    inputs = inputs.cpu()
    batch_size = min(num_images, inputs.size(0))
    grid_border_size = 2
    
    # 反標準化圖像
    images = denormalize(inputs[:batch_size])
    
    # 創建圖像網格
    grid = torchvision.utils.make_grid(images, nrow=4, padding=grid_border_size)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0))
    
    # 添加標籤
    if preds is not None:
        labels = labels[:batch_size]
        preds = preds[:batch_size]
        title = []
        for i in range(batch_size):
            title.append(f'真實: {SELECTED_CATEGORIES[labels[i]]}, 預測: {SELECTED_CATEGORIES[preds[i]]}')
        plt.title('\n'.join(title))
    else:
        labels = labels[:batch_size]
        title = []
        for i in range(batch_size):
            title.append(f'類別: {SELECTED_CATEGORIES[labels[i]]}')
        plt.title('\n'.join(title))
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"批次可視化已保存到 {save_path}")
    
    plt.show()

def plot_training_history(history, save_path=None):
    """繪製訓練歷史"""
    plt.figure(figsize=(12, 4))
    
    # 繪製損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='訓練')
    plt.plot(history['val_loss'], label='驗證')
    plt.title('損失')
    plt.xlabel('Epoch')
    plt.ylabel('損失')
    plt.legend()
    
    # 繪製準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='訓練')
    plt.plot(history['val_acc'], label='驗證')
    plt.title('準確率')
    plt.xlabel('Epoch')
    plt.ylabel('準確率')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"訓練歷史已保存到 {save_path}")
    
    plt.show()

def visualize_model_predictions(model, dataloader, device, num_images=16, save_dir=None):
    """可視化模型預測"""
    model.eval()
    images_so_far = 0
    
    # 創建保存目錄
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 定義結果追蹤字典
    results = {
        'correct': [],
        'incorrect': []
    }
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if images_so_far >= num_images:
                break
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 收集正確和錯誤的預測
            for j in range(inputs.size(0)):
                if images_so_far >= num_images:
                    break
                    
                images_so_far += 1
                
                # 獲取圖像
                img = denormalize(inputs[j].cpu())
                img = torchvision.transforms.ToPILImage()(img)
                
                # 獲取標籤
                true_label = SELECTED_CATEGORIES[labels[j].item()]
                pred_label = SELECTED_CATEGORIES[preds[j].item()]
                is_correct = (preds[j] == labels[j])
                
                # 保存結果
                result_dict = {
                    'image': img,
                    'true_label': true_label,
                    'pred_label': pred_label
                }
                
                if is_correct:
                    results['correct'].append(result_dict)
                else:
                    results['incorrect'].append(result_dict)
                
                # 保存圖像
                if save_dir:
                    status = "correct" if is_correct else "incorrect"
                    filename = f"{status}_{images_so_far}_true-{true_label}_pred-{pred_label}.png"
                    img.save(os.path.join(save_dir, filename))
    
    # 可視化一些正確的預測
    if results['correct']:
        plt.figure(figsize=(12, 8))
        plt.suptitle('正確的預測', fontsize=16)
        
        for i, result in enumerate(results['correct'][:8]):
            plt.subplot(2, 4, i + 1)
            plt.imshow(result['image'])
            plt.title(f"真實: {result['true_label']}\n預測: {result['pred_label']}")
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'correct_predictions.png'), bbox_inches='tight')
        
        plt.show()
    
    # 可視化一些錯誤的預測
    if results['incorrect']:
        plt.figure(figsize=(12, 8))
        plt.suptitle('錯誤的預測', fontsize=16)
        
        for i, result in enumerate(results['incorrect'][:8]):
            plt.subplot(2, 4, i + 1)
            plt.imshow(result['image'])
            plt.title(f"真實: {result['true_label']}\n預測: {result['pred_label']}")
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_dir:
            plt