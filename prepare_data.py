import os
import sys

# 將專案根目錄加入 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config import DATA_ROOT, COCO_PATH, SELECTED_CATEGORIES, RANDOM_SEED
import os
import json
import shutil
from pycocotools.coco import COCO
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
from config import DATA_ROOT, COCO_PATH, SELECTED_CATEGORIES, RANDOM_SEED

def download_coco():
    """下載 COCO 數據集 (如果尚未下載)"""
    if not os.path.exists(COCO_PATH):
        os.makedirs(COCO_PATH, exist_ok=True)
        print("請手動下載 COCO 2017 數據集到目錄:", COCO_PATH)
        print("下載連結: https://cocodataset.org/#download")
        print("您需要下載 '2017 Train images' 和 '2017 Val images' 以及它們的標注文件")
        print("下載後，您的資料夾結構應該像這樣:")
        print(f"{COCO_PATH}/annotations/")
        print(f"{COCO_PATH}/train2017/")
        print(f"{COCO_PATH}/val2017/")
        return False
    return True

def prepare_classification_dataset():
    """準備分類數據集，提取所選類別的圖像"""
    # 確認數據集路徑
    if not download_coco():
        return

    # 設置種子以確保可重複性
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # 初始化 COCO API
    annFile = f'{COCO_PATH}/annotations/instances_train2017.json'
    coco = COCO(annFile)
    
    # 獲取類別 ID
    cat_ids = {}
    for category in SELECTED_CATEGORIES:
        cat_ids[category] = coco.getCatIds(catNms=[category])
    
    # 創建輸出目錄
    output_dir = os.path.join(DATA_ROOT, 'coco_classification')
    for split in ['train', 'val', 'test']:
        for category in SELECTED_CATEGORIES:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)
    
    # 為每個類別處理圖像
    for category in SELECTED_CATEGORIES:
        print(f"Processing {category} images...")
        img_ids = coco.getImgIds(catIds=cat_ids[category])
        np.random.shuffle(img_ids)
        
        # 分割數據集 (70% 訓練, 15% 驗證, 15% 測試)
        train_size = int(len(img_ids) * 0.7)
        val_size = int(len(img_ids) * 0.15)
        
        train_ids = img_ids[:train_size]
        val_ids = img_ids[train_size:train_size+val_size]
        test_ids = img_ids[train_size+val_size:]
        
        # 處理訓練集
        process_image_set(coco, train_ids, category, os.path.join(output_dir, 'train', category))
        # 處理驗證集
        process_image_set(coco, val_ids, category, os.path.join(output_dir, 'val', category))
        # 處理測試集
        process_image_set(coco, test_ids, category, os.path.join(output_dir, 'test', category))
    
    print(f"Dataset prepared at {output_dir}")
    
    # 產生統計資訊
    generate_stats(output_dir)

def process_image_set(coco, img_ids, category, output_dir):
    """處理並複製指定類別的圖像到目標目錄"""
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_file = img_info['file_name']
        src_path = os.path.join(COCO_PATH, 'train2017', img_file)
        dst_path = os.path.join(output_dir, img_file)
        
        # 如果源文件存在，則複製
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

def generate_stats(dataset_dir):
    """產生數據集統計資訊"""
    stats = {'total': 0}
    
    for split in ['train', 'val', 'test']:
        stats[split] = {}
        split_total = 0
        
        for category in SELECTED_CATEGORIES:
            path = os.path.join(dataset_dir, split, category)
            count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
            stats[split][category] = count
            split_total += count
        
        stats[split]['total'] = split_total
        stats['total'] += split_total
    
    # 打印統計資訊
    print("Dataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    # 保存統計資訊到文件
    with open(os.path.join(dataset_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    prepare_classification_dataset()
