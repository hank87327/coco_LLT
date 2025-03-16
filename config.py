import os
import torch

# 基本設定
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 數據設定
DATA_ROOT = r'C:\Users\user20608\Desktop\Interview\Wiwynn_coco_classifier\datasets'
COCO_PATH = os.path.join(DATA_ROOT, 'coco')
SELECTED_CATEGORIES = ['person', 'car', 'dog', 'laptop']  # 您選擇的4個類別
IMG_SIZE = 224  # 圖像大小
BATCH_SIZE = 32

# 訓練設定
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
LR_SCHEDULER = 'step'  # 'cosine' 或 'step'
STEP_SIZE = 7
GAMMA = 0.1
NUM_WORKERS = 4
CHECKPOINT_DIR = r'C:\Users\user20608\Desktop\Interview\Wiwynn_coco_classifier\checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 模型設定
BACKBONE = 'efficientnet_b3'  # 'efficientnet_b3' 或 'resnet50'
PRETRAINED = True
USE_ATTENTION = True

# 日誌設定
LOG_DIR = r'C:\Users\user20608\Desktop\Interview\Wiwynn_coco_classifier\logs'
os.makedirs(LOG_DIR, exist_ok=True)
