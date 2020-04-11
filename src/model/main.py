import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from eda import eda
from nets import resnet,preact
from transforms import tf
import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

def train(epoch,i):
    print('\nEpoch: %d/%d' % (epoch,i))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f'Training accuracy for epoch {epoch}:{100*(correct/total)}')
    return(round(100*(correct/total),2),train_loss)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f'Testing accuracy for epoch {epoch}:{100*(correct/total)}')
    #Saves checkpoint in a new folder as shown below. Load_state_dict to resnet18 to restore weights and biases
    #Refer to below url
    #https://pytorch.org/tutorials/beginner/saving_loading_models.html 
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return(round(acc,2),test_loss)

def run_net(i):
    testacc=[]
    testloss=[]
    trainloss=[]
    trainacc=[]
    epo=[]
    for epoch in range(start_epoch, start_epoch+i):
        a1,a2=train(epoch,i)
        a3,a4=test(epoch)
        trainacc.append(a1)
        trainloss.append(a2)
        testacc.append(a3)
        testloss.append(a4)
        epo.append(epoch)
    log=pd.Dataframe(data={'Epoch':epo,'Train accuracy':trainacc,'Train Loss':trainloss,'Test accuracy':testacc,'Test loss':testloss})
    if os.path.exists(dir_path+'//logs')==False:
        os.mkdir((dir_path+'//logs'))
    log.to_csv((dir_path+'//logs'+'//traininglog.csv'))
    print(best_acc)


if __name__ == '__main__':
    #exp=input('Do you want the EDA performed on the Cifar-10 dataset? Enter:[y/n] ')
    if os.path.exists(dir_path+'//eda//image'+'//cifar10.png')==False:
        eda.eda()
    else:
        print("Details available")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'The device being used is: {device}')
    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch



    trainset = torchvision.datasets.CIFAR10(root=dir_path[:-5]+'/data', train=True, download=True, transform=tf.transform_train())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=dir_path[:-5]+'/data', train=False, download=True, transform=tf.transform_test())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = resnet.ResNet18()
    #net = preact.PreActResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    #Specify number of epochs. 200 was the highest accuracy Resnet18 (92.7) but Resnet18 for 50 epochs gave (92.5) accuracy.    
    run_net(50)
