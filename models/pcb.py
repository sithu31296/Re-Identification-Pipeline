import torch
from torch import nn 
from torchvision.models import resnet50
from torch.nn import functional as F

class Classifier(nn.Sequential):
    def __init__(self, num_classes):
        super().__init__(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )


class PCB(nn.Module):
    """Part-based Convolutional Baseline
    Paper: https://arxiv.org/abs/1711.09349
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 751, pretrained: str = None):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((6, 1))
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.ModuleList([
            Classifier(num_classes)
        for _ in range(6)])

        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)                     # 2x2048x6x1
        x = x.view(x.shape[0], x.shape[1], -1)  # 2x2048x6
        
        if self.training:
            x = self.dropout(x)  
            y = [m(x[..., i]) for i, m in enumerate(self.classifier)]
            return y, x                         # [(2, 751)x6], 2x2048x6
        else:
            return x


if __name__ == '__main__':
    from .resnet import ResNet
    model = PCB(ResNet())
    model.train()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y[0][0].shape)