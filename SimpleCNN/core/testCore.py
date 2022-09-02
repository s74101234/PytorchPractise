import torch
import numpy as np
import progressbar
def test(net, testLoader, criterion):
    net.eval()
    # test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        bar = progressbar.ProgressBar(max_value=len(testLoader))
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            # loss = criterion(outputs, targets.type(torch.float).reshape(-1, 1))
            # test_loss += loss.item()

            predicted = np.where(outputs.cpu() > 0.5, 1, 0).reshape(-1, )
            test_total += targets.size(0)
            test_correct += np.sum(predicted == targets.cpu().numpy())
            bar.update(batch_idx)

            # print(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))
        bar.finish()
        print('Test |Test Acc: %.3f |(%d/%d)'
                     % (100.*test_correct/test_total, test_correct, test_total))
