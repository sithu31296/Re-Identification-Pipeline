from .baseline import Baseline
from .pcb import PCB
from .resnet import ResNet

model_dict = {
    'baseline': Baseline,
    'pcb': PCB
}


def choose_backbone(backbone_name: str = 'resnet50'):
    name, sub_name = backbone_name.split('-')
    name, sub_name = name.lower(), sub_name.lower()
    if name == 'resnet':
        backbone = ResNet(sub_name)
    return backbone


def choose_models(num_classes, cfg):
    model_name = cfg['MODEL']['NAME'].lower()
    backbone = choose_backbone(cfg['MODEL']['BACKBONE']['NAME'])
    assert model_name in model_dict.keys(), f'Supported models: {model_dict.keys()}'
    return model_dict[model_name](backbone, num_classes, cfg['MODEL']['BACKBONE']['PRETRAIN'])