import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append('..')
from config import SELECTED_CATEGORIES

def compute_metrics(all_preds, all_labels):
    """計算分類指標"""
    # 轉換為 numpy 數組
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    
    # 計算總體指標
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 計算每個類別的準確率
    class_acc = {}
    for i, category in enumerate(SELECTED_CATEGORIES):
        # 找出屬於該類別的所有樣本
        mask = (all_labels == i)
        if np.sum(mask) > 0:
            class_acc[category] = accuracy_score(all_labels[mask], all_preds[mask])
        else:
            class_acc[category] = 0.0
    
    # 詳細報告
    report = classification_report(all_labels, all_preds, target_names=SELECTED_CATEGORIES, digits=4)
    
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_accuracy': class_acc,
        'report': report
    }
    
    return metrics

def print_metrics(metrics):
    """打印評估指標"""
    print(f"準確率: {metrics['accuracy']:.4f}")
    print(f"精確率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1 分數: {metrics['f1']:.4f}")
    
    print("\n每個類別的準確率:")
    for category, acc in metrics['class_accuracy'].items():
        print(f"  {category}: {acc:.4f}")
    
    print("\n分類報告:")
    print(metrics['report'])

def plot_confusion_matrix(metrics, save_path=None):
    """繪製混淆矩陣"""
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SELECTED_CATEGORIES,
                yticklabels=SELECTED_CATEGORIES)
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.title('混淆矩陣')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"混淆矩陣已保存到 {save_path}")
    
    plt.show()

def evaluate_model(model, dataloader, device, criterion=None):
    """評估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
    
    # 計算指標
    metrics = compute_metrics(all_preds, all_labels)
    
    # 如果有損失函數，計算平均損失
    if criterion is not None:
        metrics['loss'] = running_loss / len(dataloader.dataset)
    
    return metrics
