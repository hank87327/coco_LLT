from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import os
import PIL.Image as Image
from typing import Dict, List, Tuple
import sys
sys.path.append('..')
from config import IMG_SIZE, BATCH_SIZE, SELECTED_CATEGORIES, DATA_ROOT, NUM_WORKERS

class CocoClassificationDataset(Dataset):
    """自定義 COCO 分類數據集"""
    
    def __init__(self, root_dir: str, split: str, transform=None):
        """
        Args:
            root_dir: 數據集根目錄
            split: 'train', 'val', 或 'test'
            transform: 可選的轉換
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.classes = SELECTED_CATEGORIES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """載入所有樣本的路徑與標籤"""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, self.split, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 找不到目錄 {class_dir}")
                continue
                
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        """返回數據集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """取得指定索引的樣本"""
        img_path, class_idx = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, class_idx
        except Exception as e:
            print(f"無法載入圖像 {img_path}: {e}")
            # 如果圖像損壞，返回數據集中的下一個樣本
            return self.__getitem__((idx + 1) % len(self))

def get_transforms() -> Dict[str, transforms.Compose]:
    """獲取訓練和評估的轉換"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

def get_dataloaders() -> Dict[str, DataLoader]:
    """創建數據加載器"""
    data_dir = os.path.join(DATA_ROOT, 'coco_classification')
    transforms_dict = get_transforms()
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = CocoClassificationDataset(
            root_dir=data_dir,
            split=split,
            transform=transforms_dict[split]
        )
        
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        print(f"{split} 數據集大小: {len(dataset)}")
    
    return dataloaders

if __name__ == "__main__":
    # 測試數據加載器
    loaders = get_dataloaders()
    for split, loader in loaders.items():
        print(f"{split} 批次數: {len(loader)}")
        # 獲取第一個批次
        images, labels = next(iter(loader))
        print(f"批次形狀: {images.shape}, 標籤: {labels}")
