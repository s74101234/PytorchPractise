import torch
from core.dataLoad import dataLoadTest
from core.testCore import test

if __name__ == '__main__':
    # parameters
    height, width = 224, 224
    batchSize = 4 
    ResumeModelPath = './Model.pth'
    

    # Data load
    print('========== Preparing data. ==========')
    testLoader = dataLoadTest(batchSize, height, width)

    net = torch.load(ResumeModelPath)
    net = net.to('cuda')

    test(net, testLoader, criterion=None)