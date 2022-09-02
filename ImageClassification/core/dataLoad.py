from .customDataSet import customDataSet
from torchvision import transforms
from torch.utils.data import DataLoader

def dataLoadTrain(batchSize, height, width):
    # transforms
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=_mean, std=_std)
    ])

    transform_val = transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=_mean, std=_std)
    ])

    train_set = customDataSet(root='./Data/', split='train', transform=transform_train)
    valid_set = customDataSet(root='./Data/', split='val', transform=transform_val)

    train_loader = DataLoader(dataset=train_set, batch_size=batchSize, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=valid_set, batch_size=batchSize, shuffle=False, num_workers=2)

    return train_loader, val_loader

def dataLoadTest(batchSize, height, width):
    # transforms
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]

    transform_test = transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=_mean, std=_std)
    ])

    test_set = customDataSet(root='./Data/', split='test', transform=transform_test)

    test_loader = DataLoader(dataset=test_set, batch_size=batchSize, shuffle=True, num_workers=2)

    return test_loader
