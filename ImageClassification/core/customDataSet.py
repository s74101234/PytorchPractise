import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
class customDataSet(Dataset):
    def __init__(self, root, split, transform = None):
        
        filePath = root + split + '.csv'
        DataSet = np.genfromtxt(filePath, delimiter=',', dtype=str)
        self.root = root
        self.imgs = DataSet[0:, 0]
        self.labels = DataSet[0:, 1]
        self.transform = transform

    def __getitem__(self, index):
        imgpath = self.root + self.imgs[index]
        img = Image.open(imgpath).convert('RGB')
        label = int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
