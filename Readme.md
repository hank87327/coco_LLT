
## 項目結構

```
coco_classifier/
├── data/                  # 資料相關
│   ├── dataset.py         # 資料集類別定義
│   └── prepare_data.py    # 資料集準備腳本
├── models/                # 模型相關
│   ├── __init__.py
│   └── classifier.py      # 模型定義
├── utils/                 # 工具函數
│   ├── __init__.py
│   ├── metrics.py         # 評估指標
│   └── visualization.py   # 可視化工具
├── config.py              # 設定檔
├── train.py               # 訓練腳本
├── evaluate.py            # 評估腳本
└── README.md              # 專案說明
```

## 環境設置

```bash
# 創建並啟動虛擬環境
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 安裝依賴
pip install torch torchvision matplotlib numpy pandas seaborn scikit-learn jupyterlab pycocotools tqdm
```

## 數據準備

1. 下載 COCO 2017 數據集
   - 訓練集圖像 (train2017)
   - 驗證集圖像 (val2017)
   - 標注 (annotations)

2. 運行數據準備腳本
   ```bash
   python data/prepare_data.py
   ```

## 訓練模型

```bash
python train.py
```

這將根據 `config.py` 中的設置訓練模型，並將結果保存到 `logs` 和 `checkpoints` 目錄中。

## 評估模型

```bash
python evaluate.py
```

評估將在測試集上運行，並檢查每個類別是否達到 95% 的準確率。

## 模型架構

- **骨幹網絡**: EfficientNet B3
- **自定義修改**:
  - 加入 Squeeze-and-Excitation(SE) 注意力機制
  - 使用自定義分類頭


## 作者

[陳柏翰]
## 備註:

## 因為本身自身硬體設備不足導致沒辦法進行諸多地方的改進，倘若要進行其改善可以使用從訓練發現到的行為

## 1.可以嘗試使用FPN來處理不同尺度特徵。
## 2.對於訓練後發現LR應該使用warmup嘗試，並且提高dropout來測試是否能達到更好的效果。
## 3.類別不平衡的問題可能需要使用類別權重的CrossEntropyLoss特別給予cat&dog更高的類別權重(較容易認錯)。
## 4.然後嘗試用Focal loss來專注於個別困難樣本。
## 5.使用資料強化(特別為Random Erasing和CutMix來降低貓狗的容易錯別的現象)並且有機會提升accracy。
## 6.增加epoches的次數來提升準確率因為從訓練準確率跟驗證準確率以及曲線資料的部分整體的都還沒達到overfitting。

## 以上所敘述的這些備註及方法並不代表所有使用皆為正確，可能還是需要經過交叉測試以及訓練驗證過才能達到其最好的效果。