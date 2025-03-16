import torch
import torch.nn as nn
import torchvision.models as models
import sys
sys.path.append('..')
from config import BACKBONE, PRETRAINED, USE_ATTENTION, SELECTED_CATEGORIES

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力塊"""
    
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CocoClassifier(nn.Module):
    """COCO 圖像分類模型"""
    
    def __init__(self, num_classes=len(SELECTED_CATEGORIES), backbone=BACKBONE, 
                 pretrained=PRETRAINED, use_attention=USE_ATTENTION):
        super(CocoClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_attention = use_attention
        
        # 載入預訓練的骨幹網絡
        if backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            # 移除原有的分類器
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            # 移除原有的分類器
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的骨幹網絡: {backbone}")
        
        # 自定義分類頭 (全局池化＋分類器)
        if use_attention:
            self.attention = SEBlock(self.feature_dim)
        else:
            self.attention = None
        
        # 定義全局池化：只在必要時使用
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, self.num_classes)
        )
    
    def forward(self, x):
        # 特徵提取
        features = self.backbone(x)
        
        # 若有注意力模組且 features 為 4 維時才使用
        if self.attention is not None and features.dim() == 4:
            features = self.attention(features)
        
        # 如果 features 不是 [batch, feature_dim]，則使用 global_pool 將其轉換
        if features.dim() != 2 or features.size(1) != self.feature_dim:
            pooled = self.global_pool(features)
        else:
            pooled = features
        
        logits = self.classifier(pooled)
        return logits

def create_model(num_classes=len(SELECTED_CATEGORIES)):
    """創建模型實例"""
    model = CocoClassifier(
        num_classes=num_classes,
        backbone=BACKBONE,
        pretrained=PRETRAINED,
        use_attention=USE_ATTENTION
    )
    return model

if __name__ == "__main__":
    # 測試模型
    model = create_model()
    print(model)
    
    # 測試前向傳播
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    print(f"輸入形狀: {input_tensor.shape}, 輸出形狀: {output.shape}")
