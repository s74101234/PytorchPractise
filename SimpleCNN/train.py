import torch
from core.dataLoad import dataLoadTrain
from core.trainCore import train
from core.model import basemodel

if __name__ == '__main__':
    # Parameters
    height, width = 224, 224
    batchSize = 4
    ClassesNum = 1 # sigmoid (0 / 1)
    epochs = 10
    saveModelPath = "./Model.pth"

    trainLoader, valLoader = dataLoadTrain(batchSize, height, width)
    net = basemodel(ClassesNum=ClassesNum)
    net = net.to('cuda')

    #Loss
    criterion = torch.nn.BCELoss() # BCELoss BCEWithLogitsLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) 

    for epoch in range(0, epochs):
        train(net, epoch, trainLoader, valLoader, optimizer, criterion, saveModelPath)
