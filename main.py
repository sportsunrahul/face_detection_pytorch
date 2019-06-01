import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import load_dataset
from model import ConvNet
from opts import opts

import cv2
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


def train(model, device, train_loader, optimizer, epoch, opt):
    correct = 0
    loss_avg = 0
    acc_avg = 0
    count = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device,dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum().item()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]Loss: {:.6f}'.format(\
                epoch, batch_idx * len(data), len(train_loader.dataset), \
                        100. * batch_idx / len(train_loader), loss.item())\
                                + ";     Accuracy: {:.3f}".format(correct/(opt.batch_size*(batch_idx+1))), end='\r')
        loss_avg  = loss_avg + loss.item()
        acc_avg = acc_avg + correct/(opt.batch_size*(batch_idx+1))
        count = count + 1

    print()
    loss_avg = loss_avg/count
    acc_avg = acc_avg/count 
    return loss_avg, acc_avg
    
def test(model, device, test_loader, mode):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device,dtype=torch.long)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(\
            mode,test_loss, correct, len(test_loader.dataset), 100.*correct / len(test_loader.dataset)))

    return test_loss, 100.*correct / len(test_loader.dataset)



def main(opt):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    train_set, val_set, test_set = load_dataset(opt)
    train_loader = data_utils.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,  num_workers=4)
    val_loader = data_utils.DataLoader(val_set, batch_size=opt.batch_size, shuffle=True,  num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batch_size, shuffle=True, num_workers=4)


    model = ConvNet().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2)
    optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    train_loss_list, train_acc_list = [],[]
    val_loss_list, val_acc_list = [],[]

    for epoch in range(1, opt.epochs + 1):
        loss, acc = train(model, device, train_loader, optimizer, epoch, opt)
        train_loss_list.append(loss)
        train_acc_list.append(acc)
        # print(train_loss_list[-1], train_acc_list[-1])
        if(epoch % opt.val_intervals == 0):
                loss, acc = test(model, device, val_loader, 'Validation')
                val_loss_list.append(loss)
                val_acc_list.append(acc)
                # print(val_loss_list[-1], val_acc_list[-1])


    
    test(model, device, test_loader, 'Test')
    plt.figure(1)
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(['Train','Validation'])
    plt.title('Loss Plot')
    plt.savefig("loss.png")
    plt.show()

    plt.figure(2)
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.ylabel("accuarcy")
    plt.xlabel("epochs")
    plt.legend(['Train','Validation'])
    plt.title('Accuracy Plot')
    plt.savefig("accuracy.png")
    plt.show()


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)