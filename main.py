import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms

from model import Net

def main():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net(num_rej=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, hidden1, hidden2 = net(inputs)
            #print(hidden1)
            #print(hidden2)
            loss = criterion(hidden1, labels)
            loss = net.pconf(hidden2, labels, loss, rate_all=0.3, rate_inv=0.8)
            #print(loss.data[0])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            #print(loss.data[0])
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'reject')

    outputs, h1, h2 = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    num_rej = 0
    for data in testloader:
        images, labels = data
        outputs, h1, h2 = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        re = (predicted == 10).sum()
        num_rej += re
        correct += re

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print('The percent of objects rejected: %.3f %%' % (
        100 * num_rej / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs, h1, h2 = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        rej = (predicted == 10)
        for i in range(4):
            if rej[i] != 1:
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1
            else:
                print(labels[i])
                print(outputs[i])


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    main()