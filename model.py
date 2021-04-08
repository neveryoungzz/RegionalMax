#-*- coding:utf- -*-
import numpy as np
import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import TestLabelGenerate
from configargparse import ArgParser

class Torch_Dataset(torch.utils.data.Dataset):
    def __init__(self, arr):
        self.arr = arr
    
    def __len__(self):
        return self.arr.shape[0]
    
    def __getitem__(self, idx):
        return self.arr[idx]

class LSTMModel(nn.Module):
    def __init__(self, numEmbed = 10, seqLen = 10, dimEmbed = 8, 
                 numLayers = 1, isBidirect = False,
                 dropout = 0, hidVar = 30):
        super(LSTMModel, self).__init__()
        self.numEmbed = numEmbed
        self.seqLen = seqLen
        self.dimEmbed = dimEmbed
        self.numLayers = numLayers
        self.isBidirect = isBidirect
        if self.isBidirect:
            self.directions = 2
        else:
            self.directions = 1
        self.dropout = dropout
        self.hidVar = hidVar
        self.seqEmbed = nn.Embedding(self.numEmbed, self.dimEmbed)
        self.scalarEmbed = nn.Embedding(self.numEmbed, self.dimEmbed)
        self.lstm = nn.LSTM(input_size = self.dimEmbed, hidden_size = self.hidVar, 
                            num_layers = self.numLayers, batch_first = True,
                            bidirectional = self.isBidirect, dropout = self.dropout)
        self.linear = nn.Linear(self.hidVar, self.numEmbed)

    def forward(self, inputData):
        sequence = inputData[:, :-2]
        seqEmbeded = self.seqEmbed(sequence)
        scalar = inputData[:, -2]
        sclrEmbeded = self.scalarEmbed(scalar.sub(1))
        h_0 = torch.cat([sclrEmbeded.reshape(-1, self.dimEmbed).float(),
                         torch.zeros(inputData.size(0), self.hidVar - self.dimEmbed, dtype = torch.float)]
                        , dim = 1).reshape(1, -1, self.hidVar).repeat((self.numLayers * self.directions, 1, 1))
        c_0 = torch.zeros(self.numLayers * self.directions, 
                          inputData.size(0), 
                          self.hidVar, dtype = torch.float)
        y_t, (h_t, c_t) = self.lstm(seqEmbeded.float(), (h_0, c_0))
        return self.linear(c_t[0, :, :])
    
def train(epoch, data_loader):
    model.train()
    train_ce = 0
    correct = 0
    for batch_idx, data in enumerate(data_loader):
        label = data[:, -1]
        optimizer.zero_grad()
        label_score = model(data)
        var = Variable(data)
        ce = crossEntropy(label_score.view(32, 10).float(), label.long())
        ce.backward()
        train_ce += ce.data
        optimizer.step()
        batch_correct = torch.sum(torch.argmax(label_score.view(-1, 10), dim = 1) == label)
        correct += batch_correct.data
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCross Entropy: {:.6f}\tCorrectly predicted data: {:d}/{}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), ce.data, 
                    batch_correct.data, batchSize))
    acc = correct.float().data / batchSize / batch_idx
    print('====> Epoch: {} Average Cross Entropy: {:.4f} Accuracy: {:.4f}'.format(
              epoch, ce, acc))
    return acc

def evaluation(data_loader):
    model.eval()
    correct = 0
    for idx, data in enumerate(data_loader):
        label = data[:, -1]
        label_score = model(data)
        batch_correct = torch.sum(torch.argmax(label_score.view(-1, 10), dim = 1) == label)
        correct += batch_correct.data
    acc = correct.float().data / idx / batchSize
    print ('Test Accuracy: {:.5f}'.format(acc.data))
    return acc

def draw(data, color):
    plt.plot(np.arange(len(data)), data, color, linewidth = 2.0)
    
if __name__ == '__main__':
    p = ArgParser()
    p.add('-i', '--input-path', help = 'Input Path', default = 'data')
    p.add('--batch-size', help = 'Batch Size', default = 32, type = int)
    p.add('--dim-embed', help = 'Embedding Dimension', default = 8, type = int)
    p.add('--num-layers', help = 'Hidden layers in LSTM model', default = 1, type = int)
    p.add('--bidirect', help = 'Is Bidirectional LSTM model (default = false)', 
           default = False, action = 'store_true')
    p.add('--dropout', help = 'Use dropout (default = false)', default = 0)
    p.add('--hidvar', help = 'Hidden Units in LSTM', default = 30, type = int)
    p.add('--lr', help = 'learing rate', default = 0.01)
    p.add('--steps', help = 'steps to run', default = 50, type = int)
    
    options = p.parse_args()
    print(options)
    inputPath = options.input_path
    batchSize = options.batch_size
    dimEmbed = options.dim_embed
    numLayers = options.num_layers
    isBidirect = options.bidirect
    dropOut =options.dropout
    hidVar = options.hidvar
    lr = options.lr
    steps = options.steps
    if not os.path.exists(os.path.join(inputPath, 'test_label.csv')):
        TestLabelGenerate(os.path.join(inputPath, 'test_seq.csv'),
                          os.path.join(inputPath, 'test_scalar.csv'),
                          os.path.join(inputPath, 'test_label.csv'))
    sequence = np.loadtxt(os.path.join(inputPath, 'sequence.csv'), 
                          delimiter = ',', dtype = int)
    scalar = np.loadtxt(os.path.join(inputPath, 'scalar.csv'), 
                        delimiter = ',', dtype = int)
    label = np.loadtxt(os.path.join(inputPath, 'label.csv'), 
                       delimiter = ',', dtype = int)
    test_seq = np.loadtxt(os.path.join(inputPath, 'test_seq.csv'), 
                          delimiter = ',', dtype = int)
    test_scalar = np.loadtxt(os.path.join(inputPath, 'test_scalar.csv'), 
                             delimiter = ',', dtype = int)
    test_label = np.loadtxt(os.path.join(inputPath, 'test_label.csv'), 
                            delimiter = ',', dtype = int)
    data = np.column_stack([sequence, scalar, label])
    test_data = np.column_stack([test_seq, test_scalar, test_label])
    print('Data loaded, train data shape {}, test data shape {}'.format(data.shape, test_data.shape))

    loaderKwargs = {'num_workers': 0, 'pin_memory': True, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(Torch_Dataset(data), batch_size = batchSize, **loaderKwargs)
    test_loader = torch.utils.data.DataLoader(Torch_Dataset(test_data), batch_size = batchSize, **loaderKwargs)

    model = LSTMModel()
    model = model.float()
    crossEntropy = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    accs = []
    testAccs = []
    for step in range(1, steps + 1):
        accs.append(train(step, train_loader).numpy())
        testAccs.append(evaluation(test_loader).numpy())
    
    plt.figure(figsize = (10, 10))
    plt.title('Model Accuracy', fontsize=30)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    draw(accs, 'r')
    draw(testAccs, 'g')
    plt.savefig(os.path.join(inputPath, 'acc.pdf'))
    plt.close()

    