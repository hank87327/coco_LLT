import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.classifier import create_model
from utils.metrics import evaluate_model, print_metrics, plot_confusion_matrix
from utils.visualization import visualize_model_predictions
from config import DEVICE, CHECKPOINT_DIR, LOG_DIR, SELECTED_CATEGORIES

def evaluate_best_model():
    """評估最佳模型"""
    # 載入數據
    print("載入數據...")
    dataloaders = get_dataloaders()
    
    # 創建模型
    model = create_model()
    model = model.to(DEVICE)
    
    # 載入最佳模型權重
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"載入最佳模型，Epoch: {checkpoint['epoch']+1}, 驗證準確率: {checkpoint['val_acc']:.4f}")
    else:
        print(f"錯誤：找不到模型 {best_model_path}")
        return
    
    # 設置損失函數
    criterion = nn.CrossEntropyLoss()
    
    # 評估模型
    print("在測試集上評估模型...")
    test_metrics = evaluate_model(model, dataloaders['test'], DEVICE, criterion)
    
    # 打印測試指標
    print("\n--- 測試集結果 ---")
    print_metrics(test_metrics)
    
    # 檢查每個類別是否達到 95% 的準確率
    print("\n每個類別的準確率:")
    all_above_threshold = True
    for category, acc in test_metrics['class_accuracy'].items():
        status = "通過" if acc >= 0.95 else "未通過"
        print(f"  {category}: {acc:.4f} ({status})")
        if acc < 0.95:
            all_above_threshold = False
    
    print(f"\n整體準確率: {test_metrics['accuracy']:.4f}")
    if all_above_threshold:
        print("所有類別準確率均達到 95% 或更高！")
    else:
        print("某些類別未達到 95% 的準確率要求。")
    
    # 可視化混淆矩陣
    cm_path = os.path.join(LOG_DIR, 'test_confusion_matrix.png')
    plot_confusion_matrix(test_metrics, save_path=cm_path)
    
    # 可視化一些預測
    results_dir = os.path.join(LOG_DIR, 'test_predictions')
    os.makedirs(results_dir, exist_ok=True)
    visualize_model_predictions(model, dataloaders['test'], DEVICE, num_images=32, save_dir=results_dir)
    
    # 保存詳細結果到 JSON
    result_dict = {
        'overall_accuracy': float(test_metrics['accuracy']),
        'precision': float(test_metrics['precision']),
        'recall': float(test_metrics['recall']),
        'f1': float(test_metrics['f1']),
        'class_accuracy': {k: float(v) for k, v in test_metrics['class_accuracy'].items()},
        'confusion_matrix': test_metrics['confusion_matrix'].tolist()
    }
    
    with open(os.path.join(LOG_DIR, 'test_results.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"測試結果已保存到 {os.path.join(LOG_DIR, 'test_results.json')}")
    
    return test_metrics

if __name__ == "__main__":
    evaluate_best_model()
