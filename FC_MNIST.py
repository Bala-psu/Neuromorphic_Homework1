import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import numpy as np
import time
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from enum import Enum
from lenet5 import lenet5
import math


def modified_sigmoid(x):
    sig = 1 / (1 + torch.exp(-2*x)) #Making the slope a little bit sharper
    return sig


class MSigmoid(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, input):
        return modified_sigmoid(input)

class MNIST_fc(nn.Module):
    def __init__(self):
        super(MNIST_fc, self).__init__()
        self.fclayer1 = nn.Linear(784, 100, bias = False)
        self.act = MSigmoid()
        self.fclayer2 = nn.Linear(100, 10, bias = False)
        
    def forward(self, x):
        x = x.view(-1,784)
        output = self.fclayer1(x)
        output = self.act(output)
        output = self.fclayer2(output)
        return output
  



parser = argparse.ArgumentParser(description = "Neuromorphic Computing Assignment")
parser.add_argument(
    '--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
parser.add_argument(
    '--test-batch-size', type=int, default=10000,  help='input batch size for testing (default: 10000)')
parser.add_argument('--epochs', type=int, default=20,  help='number of epochs to train (default: 30)')
parser.add_argument('--nTrials', type=int, default=1, help='number of training trials (default: 20)')
parser.add_argument('--start-lr', type=float, default=0.01,  help='Initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9,  help='momentum term used in gradient descent (default: 0.9)')
parser.add_argument('--cpu', action='store_true', default=False, help='disables CUDA training and runs training on CPU')
parser.add_argument('--log-interval', type=int, default=5,  help='how many batches to wait before logging training status')

TEST_acc = []

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(args, model, device, train_loader,optimizer, epoch, batch_size, label_features = 100):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],#batch_time, data_time,
        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    running_loss, total, correct, acc = 0,0,0,0
    train_accu, train_losses, run_acc = [], [], []
    
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        data, target = data.to(device), target.to(device)
        #target = torch.zeros(target.shape[0], label_features, dtype=torch.long, device=device).scatter_(1, target.unsqueeze(1), 1.0)
        loss = 0
        
        output = model(data)
        loss += F.cross_entropy(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))
        running_loss += loss.item()
    
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        acc = 100*correct/total
        run_acc.append(acc)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.log_interval == 0:  
            progress.display(batch_idx)
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLOSS : {}'.format(
                #epoch, batch_idx * len(data), len(train_loader.dataset),
                #100. * batch_idx / len(train_loader), loss))
    train_loss=running_loss/len(train_loader)
    accu=100.*correct/total
    train_accu.append(accu)
    train_losses.append(train_loss)
    print(len(run_acc))
    ACCURACY = run_acc
    np.save(os.path.join('/home/mdl/bus44/Neuromorphic Computing/runs/Q1b/', 'TRAIN_acc_'+ str(epoch)), ACCURACY)
            
def test(args, model, device, test_loader,data_name):
    model.eval()
    test_loss = 0
    correct = 0
    global TEST_acc
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\n{} : Average loss: {}, Accuracy: {}/{} ({}%)\n'.format(data_name,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if data_name == ' test set ':
        TEST_acc.append(test_acc)
    return correct

def save_results(state, filename='run_results'):
    directory = "runs/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


def main():
    args = parser.parse_args()
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(cuda_device)
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root = './data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.batch_size, shuffle=True,**kwargs)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root = './data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.test_batch_size, shuffle=False,**kwargs)
    
    test_performance = []
    train_performance = []
    for trial in range(1,args.nTrials+1):
        print('In trial {} of {}'.format(trial,args.nTrials))
        model = MNIST_fc().to(device)
        #model = lenet5().to(device)
        
        
        if args.cuda:
            model.cuda()
        print(model)

        optimizer = optim.SGD(filter(lambda x : x.requires_grad,model.parameters()), nesterov = True,lr=args.start_lr, momentum=0.9, weight_decay=0.0)        

        test_performance.append([])
        train_performance.append([])        
        for epoch in range(1, args.epochs + 1):            
            print('learning rate: {}'.format(optimizer.param_groups[0]['lr']))
            train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
            correct_test = test(args, model, device, test_loader,' test set ' )
            correct_train = test(args, model, device, train_loader,' train set ' ) 

            test_performance[-1].append(correct_test)
            train_performance[-1].append(correct_train)            
            save_results({'test' : test_performance,'train' : train_performance,'model_desc': repr(model),'state_dict' : model.state_dict()},filename = 'Q1b_results')
        np.save(os.path.join('/home/mdl/bus44/Neuromorphic Computing/runs/Q1b/', 'TEST_acc_'+ str(epoch)), TEST_acc)

            
if __name__ == '__main__':
    main()  