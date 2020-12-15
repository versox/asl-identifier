import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config

transform = transforms.Compose(
    [transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root=config.dataRoot + 'asl_alphabet_train/', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root=config.dataRoot + 'asl_alphabet_test/', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    #train on GPU instead of CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    """
        For squeezenet:
        net = models.squeezenet1_0(pretrained=True)
        net.classifier[1] = nn.Conv2d(512, 29, kernel_size=(1,1), stride=(1,1))
        For alexnet:
        net = models.alexnet(pretrained=True)
        net.classifier[6] = nn.Linear(4096, 29)
        For resnet18:
        net = models.resnet18(pretrained=True)
        net.fc = nn.Linear(512, 29)
    """
    
    net = models.squeezenet1_0(pretrained=True)
    net.classifier[1] = nn.Conv2d(512, 29, kernel_size=(1,1), stride=(1,1))

    #train on GPU instead of CPU
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #GPU, not CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:    # print every mini-batch
                print('[%d, %5d] loss: %.15f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    
    #*** Change name of saved model if using different pretrained ***
    PATH = './ASL_alphabet.pth'

    torch.save(net.state_dict(), PATH)