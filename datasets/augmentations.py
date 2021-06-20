from torchvision import transforms

def get_transforms(cfg):
    train_transform = transforms.Compose([
        transforms.Resize(cfg['TRAIN']['IMG_SIZE']),
        transforms.ColorJitter(cfg['TRAIN']['AUG']['B_P'], cfg['TRAIN']['AUG']['C_P'], cfg['TRAIN']['AUG']['S_P'], cfg['TRAIN']['AUG']['H_P']),
        transforms.RandomGrayscale(cfg['TRAIN']['AUG']['G_P']),   # Local Grayscale Transformation https://arxiv.org/abs/2101.08533
        transforms.Pad(10),                                 
        transforms.RandomCrop(cfg['TRAIN']['IMG_SIZE']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(cfg['TRAIN']['AUG']['RE_P']),                                        # Random Erasing Data Augmentation https://arxiv.org/pdf/1708.04896
    ])

    test_transform = transforms.Compose([
        transforms.Resize(cfg['EVAL']['IMG_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform