from __future__ import print_function
import argparse
import multiprocessing
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    total_batch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output.data, dim=1)
        total_acc += (predicted == target).sum().item() / len(target)
        total_batch += 1
    training_loss = total_loss / total_batch
    training_acc = total_acc / total_batch
    return training_acc, training_loss


def test(model, device, test_loader):
    model.eval()
    testing_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            testing_loss += loss.item()
            _, predicted = torch.max(output.data, dim=1)
            correct += (predicted == target).sum().item()
        testing_loss = testing_loss / len(test_loader.dataset)
        testing_acc = correct / len(test_loader.dataset)
    return testing_acc, testing_loss


def plot(epoches, performance):
    label = performance.pop(0)
    plt.title(label)
    plt.plot(epoches, performance, label=label)
    plt.xlabel('epoches')
    plt.legend()
    plt.savefig(label + '.jpg')
    plt.show()


def run(config, processname):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()
    seed = -1
    if processname == 1:
        seed = config.seed1
        print('First Process Start. Random Seed is', seed)
    if processname == 2:
        seed = config.seed2
        print('Second Process Start. Random Seed is', seed)
    if processname == 3:
        seed = config.seed3
        print('Third Process Start. Random Seed is', seed)
    torch.manual_seed(seed)
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []
    if processname == 1:
        training_accuracies.append('Process 1 Training Accuracy')
        training_loss.append('Process 1 Training Loss')
        testing_accuracies.append('Process 1 Testing Accuracy')
        testing_loss.append('Process 1 Testing Loss')
        training_file = open('process 1 training data.txt', mode='a')
        testing_file = open('process 1 testing data.txt', mode='a')
    elif processname == 2:
        training_accuracies.append('Process 2 Training Accuracy')
        training_loss.append('Process 2 Training Loss')
        testing_accuracies.append('Process 2 Testing Accuracy')
        testing_loss.append('Process 2 Testing Loss')
        training_file = open('process 2 training data.txt', mode='a')
        testing_file = open('process 2 testing data.txt', mode='a')
    elif processname == 3:
        training_accuracies.append('Process 3 Training Accuracy')
        training_loss.append('Process 3 Training Loss')
        testing_accuracies.append('Process 3 Testing Accuracy')
        testing_loss.append('Process 3 Testing Loss')
        training_file = open('process 3 training data.txt', mode='a')
        testing_file = open('process 3 testing data.txt', mode='a')
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        epoches.append(epoch)
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        train_acc, train_loss = 0.0, 0.0
        test_acc, test_loss = test(model, device, test_loader)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        test_acc, test_loss = 0.0, 0.0
        scheduler.step()
        training_file.writelines([str(training_loss[epoch]), ' ', str(training_accuracies[epoch]), '\n'])
        testing_file.writelines([str(testing_loss[epoch]), ' ', str(testing_accuracies[epoch]), '\n'])
        print('[ Process', processname, 'epoch', epoch, ']')
        print('Training Loss:', training_loss[epoch], 'Training Accuracy:', (100 * training_accuracies[epoch]),
              '%')
        print('Testing Loss:', testing_loss[epoch], 'Testing Accuracy:', 100 * testing_accuracies[epoch], '%')
        print('')
    plot(epoches, training_loss)
    plot(epoches, training_accuracies)
    plot(epoches, testing_accuracies)
    plot(epoches, testing_loss)
    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot_mean():
    Process_1_training_data = []
    Process_2_training_data = []
    Process_3_training_data = []
    Process_1_testing_data = []
    Process_2_testing_data = []
    Process_3_testing_data = []
    Mean_training_loss = []
    Mean_testing_loss = []
    Mean_training_accuracy = []
    Mean_testing_accuracy = []
    Epoches = []
    with open('process 1 training data.txt', mode='a', encoding='utf-8') as trainingfile1:
        with open('process 1 training data.txt', mode='r', encoding='utf-8') as traininginfile1:
            for line in traininginfile1:
                dataline = line.strip("\n").split()
                Process_1_training_data.append([float(j) for j in dataline])
    with open('process 2 training data.txt', mode='a', encoding='utf-8') as trainingfile2:
        with open('process 2 training data.txt', mode='r', encoding='utf-8') as traininginfile2:
            for line in traininginfile2:
                dataline = line.strip("\n").split()
                Process_2_training_data.append([float(j) for j in dataline])
    with open('process 3 training data.txt', mode='a', encoding='utf-8') as trainingfile3:
        with open('process 3 training data.txt', mode='r', encoding='utf-8') as traininginfile3:
            for line in traininginfile3:
                dataline = line.strip("\n").split()
                Process_3_training_data.append([float(j) for j in dataline])
    with open('process 1 testing data.txt', mode='a', encoding='utf-8') as testingfile1:
        with open('process 1 testing data.txt', mode='r', encoding='utf-8') as testinginfile1:
            for line in testinginfile1:
                dataline = line.strip("\n").split()
                Process_1_testing_data.append([float(j) for j in dataline])
    with open('process 2 testing data.txt', mode='a', encoding='utf-8') as testingfile2:
        with open('process 2 testing data.txt', mode='r', encoding='utf-8') as testinginfile2:
            for line in testinginfile2:
                dataline = line.strip("\n").split()
                Process_2_testing_data.append([float(j) for j in dataline])
    with open('process 3 testing data.txt', mode='a', encoding='utf-8') as testingfile3:
        with open('process 3 testing data.txt', mode='r', encoding='utf-8') as testinginfile3:
            for line in testinginfile3:
                dataline = line.strip("\n").split()
                Process_3_testing_data.append([float(j) for j in dataline])
    for k in range(len(Process_1_training_data)):
        Epoches.append(k + 1)
        Mean_training_loss.append(
            (Process_1_training_data[k][0] + Process_2_training_data[k][0] + Process_3_training_data[k][0]) / 3)
        Mean_testing_loss.append(
            (Process_1_testing_data[k][0] + Process_2_testing_data[k][0] + Process_3_testing_data[k][0]) / 3)
        Mean_training_accuracy.append(
            (Process_1_training_data[k][1] + Process_2_training_data[k][1] + Process_3_training_data[k][1]) / 3)
        Mean_testing_accuracy.append(
            (Process_1_testing_data[k][1] + Process_2_testing_data[k][1] + Process_3_testing_data[k][1]) / 3)
    plt.title('Mean training accuracies')
    plt.plot(Epoches, Mean_training_accuracy, label='Mean training accuracies')
    plt.xlabel('epoches')
    plt.legend()
    plt.savefig('Mean training accuracies.jpg')
    plt.show()
    plt.title('Mean testing accuracies')
    plt.plot(Epoches, Mean_testing_accuracy, label='Mean testing accuracies')
    plt.xlabel('epoches')
    plt.legend()
    plt.savefig('Mean testing accuracies.jpg')
    plt.show()
    plt.title('Mean training loss')
    plt.plot(Epoches, Mean_training_loss, label='Mean training loss')
    plt.xlabel('epoches')
    plt.legend()
    plt.savefig('Mean training loss.jpg')
    plt.show()
    plt.title('Mean testing loss')
    plt.plot(Epoches, Mean_testing_loss, label='Mean testing loss')
    plt.xlabel('epoches')
    plt.legend()
    plt.savefig('Mean testing loss.jpg')
    plt.show()


if __name__ == '__main__':
    arg = read_args()
    config = load_config(arg)
    file1 = open('process 1 training data.txt', 'w').close()
    file2 = open('process 1 testing data.txt', 'w').close()
    file3 = open('process 2 training data.txt', 'w').close()
    file4 = open('process 2 testing data.txt', 'w').close()
    file5 = open('process 3 training data.txt', 'w').close()
    file6 = open('process 3 testing data.txt', 'w').close()
    print("Multiprocessing")
    processes = []
    for i in range(1, 4):
        p = multiprocessing.Process(target=run, args=(config, i))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    plot_mean()
