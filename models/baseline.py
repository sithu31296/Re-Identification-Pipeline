import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Baseline(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 751, pretrained: str = None):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(2048)
        self.classifier = nn.Linear(2048, num_classes)

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

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)                
        x = self.avgpool(x)                 # bx2048x1x1
        x = x.view(x.shape[0], -1)          # bx2048
        y = self.bn(x)                      # bx2048

        if self.training:
            return self.classifier(y), x    # bxnc, bx2048
        return y                            # bx2048

if __name__ == '__main__':
    model = Baseline()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)