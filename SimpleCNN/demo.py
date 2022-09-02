import torch
from torchvision import transforms
from core.testCore import test
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # parameters
    height, width = 224, 224
    batchSize = 4 
    ResumeModelPath = './Model.pth'
    inputImg = './input.jpg'
    

    # Data load
    print('========== Preparing data. ==========')
    img = Image.open(inputImg).convert('RGB')
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]
    transform_demo = transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=_mean, std=_std)
    ])
    inputs = transform_demo(img)
    inputs = inputs.reshape(1, inputs.shape[0], inputs.shape[1], inputs.shape[2])

    print('========== Predict. ==========')
    net = torch.load(ResumeModelPath)
    net = net.to('cuda')
    net.eval()
    with torch.no_grad():
        outputs = net(inputs.to('cuda'))
        predicted = np.where(outputs.cpu() > 0.5, 1, 0).reshape(-1, )
        print('predict：', predicted)
