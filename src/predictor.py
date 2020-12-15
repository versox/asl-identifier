import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((200, 200)),
    # transforms.RandomHorizontalFlip(p=1.0), # randomly flip the image 100% of the time :D
    #   ^ moved this into video so it can actually be seen
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space')

torch.multiprocessing.freeze_support()

PATH = './ASL_alphabet.pth'
net = models.squeezenet1_0(pretrained=True)
net.classifier[1] = nn.Conv2d(512, 29, kernel_size=(1,1), stride=(1,1))
net.load_state_dict(torch.load(PATH))
net.eval()

def get_prediction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img.unsqueeze_(0)  # adds an extra dim at front (index 0), thus turning img into mini batch
    with torch.no_grad():
        output = net(img)
        _, predicted = torch.max(output.data, 1)
    return classes[predicted[0]]
