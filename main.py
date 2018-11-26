from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data_loader import DataLoader
import numpy as np

'''
    Generator class - consisting of 3 generators each of sizes 9, 4 and 1
'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.T1 = nn.Conv2d(512, 9, kernel_size=3)
        self.T2 = nn.Conv2d(512, 4, kernel_size=3)
        self.T3 = nn.Conv2d(512, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
    def forward(self, x):
        
        T1_out = F.relu(self.T1(x))
        S2 = F.relu(self.pool(x))
        T2_out = F.relu(self.T2(S2))
        S3 = F.relu(self.pool(S2))
        T3_out = F.relu(self.T3(S3))
        
        return T1_out, T2_out, T3_out

'''
    Binary classification after the contrastive convolution
'''
class BinaryClassification(nn.Module):
    def __init__(self, n):
        super(BinaryClassification, self).__init__()
        self.fc1 = nn.Linear(n, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x
'''
    Regressor for the kernel - H(K)
'''
class Regressor(nn.Module):
    def __init__(self, n):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(n, 1)
        
    def forward(self, x):
        x = x.view(-1, 769*11*11)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(x)
        return x
'''
    Main network consisting of 4 conv and 1 contrastive conv layer
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        self.gen = Generator()
        self.binary = BinaryClassification(769)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))

        T1, T2, T3 = self.gen(x)
        T1_flat = T1.view(-1, 9*9*9)
        T2_flat = T2.view(-1, 4*3*3)
        T3_flat = T3.view(-1, 1*2*2)
        kernel = torch.cat((T1_flat, T2_flat, T3_flat), dim=1)
        binary = self.binary(kernel)
        
        return x, kernel, F.softmax(binary)
'''
Main train method
'''
def train(args, model, reg_model, device, train_loader, optimizer, epoch):
    model.train()
    reg_model.train()
    for batch_idx, (data_1, data_2, target) in enumerate(train_loader):
        data_1, data_2, target = (data_1).to(device), (data_2).to(device), torch.from_numpy(np.asarray(target)).to(device)
        target = target.float()
        optimizer.zero_grad()
        
        F1, output_1, binary = model(data_1)
        F2, output_2, _ = model(data_2)
        Kab = torch.abs(output_1 - output_2)  # B x 769
        output_1 = torch.unsqueeze(output_1, 1) # B x 1 x 769
        output_1 = output_1.expand(output_1.size()[0], 512, 769) # B x 512 x 769
        output_1 = torch.transpose(output_1, 1, 2)

        output_2 = torch.unsqueeze(output_2, 1) # B x 1 x 769
        output_2 = output_2.expand(output_2.size()[0], 512, 769) # B x 512 x 769
        output_2 = torch.transpose(output_2, 1, 2)

        A_list = torch.Tensor()
        B_list = torch.Tensor()

        for i in range(Kab.size()[0]):
            kernel_A = output_1[i]
            kernel_A = kernel_A.reshape((kernel_A.size()[0], kernel_A.size()[1], 1, 1))
            F1_indi = F1[i]

            F1_indi = F1_indi.reshape(1, F1_indi.size()[0], F1_indi.size()[1], F1_indi.size()[2])
            F_A_B = F.conv2d(F1_indi, kernel_A)
            A_list = torch.cat((A_list, F_A_B))

            # kernel convolution for B
            kernel_B = output_2[i]
            kernel_B = kernel_B.reshape((kernel_B.size()[0], kernel_B.size()[1], 1, 1))
            F2_indi = F2[i]

            F2_indi = F2_indi.reshape(1, F2_indi.size()[0], F2_indi.size()[1], F2_indi.size()[2])
            F_B_A = F.conv2d(F2_indi, kernel_B)
            B_list = torch.cat((B_list, F_B_A))

        reg_1 = reg_model(A_list)
        reg_2 = reg_model(B_list)
        SAB = (reg_1 + reg_2)/2.0
        S_AB = torch.tensor((1-SAB))
        S_AB = torch.cat((S_AB, SAB), dim=1)
        
        loss = nn.BCELoss()(binary, target) + nn.BCELoss()(S_AB, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_loader = DataLoader(data_path='./lfw-deepfunneled', transform=data_transform, trainval="./pairsDevTrain.txt")
    train_loader = torch.utils.data.DataLoader(dataset=dataset_loader,
                                               batch_size=args.batch_size, 
                                               shuffle=True, num_workers=4)
    model = Net().to(device)
    reg_model = Regressor(769*11*11).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)


if __name__ == '__main__':
    main()