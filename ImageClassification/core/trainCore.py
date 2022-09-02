import torch
import numpy as np
import progressbar # pip install progressbar2
def train(net, epoch, trainLoader, valLoader, optimizer, criterion, saveModelPath):
    # ==================================================
    # Train
    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    bar = progressbar.ProgressBar(max_value=len(trainLoader))
    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets.type(torch.float).reshape(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        predicted = np.where(outputs.cpu() > 0.5, 1, 0).reshape(-1, )
        train_total += targets.size(0)
        train_correct += np.sum(predicted == targets.cpu().numpy())

        bar.update(batch_idx)
        # print(batch_idx, len(trainLoader), 'Loss: %.3f | Acc: %.3f (%d/%d)'% 
        #     (train_loss/(batch_idx+1), 100.*train_correct/train_total, train_correct, train_total))
    bar.finish()
    trainAcc = 100.*train_correct/train_total 
    trainLoss = train_loss/train_total
    print('Train Loss: %.3f |Train Acc: %.3f |(%d/%d)'
         % (trainLoss, trainAcc, train_correct, train_total)) 
    # ==================================================
    # Validation
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        bar = progressbar.ProgressBar(max_value=len(valLoader))
        for batch_idx, (inputs, targets) in enumerate(valLoader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets.type(torch.float).reshape(-1, 1))
            val_loss += loss.item()
            predicted = np.where(outputs.cpu() > 0.5, 1, 0).reshape(-1, )
            val_total += targets.size(0)
            val_correct += np.sum(predicted == targets.cpu().numpy())
            bar.update(batch_idx)
            # print(batch_idx, len(valLoader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
            #         %(val_loss/(batch_idx+1), 100.*val_correct/val_total, val_correct, val_total))
        bar.finish()
    valAcc = 100.*val_correct/val_total 
    valLoss = val_loss/val_total
    print('Val Loss: %.3f |Val Acc: %.3f |(%d/%d)'
         % (val_loss, valAcc, val_correct, val_total)) 
    # ==================================================
    torch.save(net, saveModelPath)
