from .loss import CrossEntropyLoss, TripletLoss, CircleLoss, CombineLoss

__all__ = ['ce', 'triplet', 'circle', 'ce+triplet']

def choose_loss(loss_fn_name: str, num_classes: int, cfg):
    assert loss_fn_name in __all__, f'Supported Loss Function Names: {__all__}'
    return {
        'ce': CrossEntropyLoss(num_classes, cfg['TRAIN']['LOSS']['EPSILON']),
        'triplet': TripletLoss(cfg['TRAIN']['LOSS']['MARGIN']),
        'circle': CircleLoss(cfg['TRAIN']['LOSS']['MARGIN'], cfg['TRAIN']['LOSS']['GAMMA']),
        'ce+triplet': CombineLoss(num_classes, cfg['TRAIN']['LOSS']['EPSILON'], cfg['TRAIN']['LOSS']['MARGIN'])
    }[loss_fn_name]
