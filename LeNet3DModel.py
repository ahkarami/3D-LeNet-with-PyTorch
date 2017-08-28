# Besmei Taala
# Amir Hossein Karami
# Subject: I want to make the LeNet Model 3D


from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


# input: (N = batch_size, C = 3, D = 32, H = 32, W = 32)
# output: (N, num_classes)
num_classes = 10


class LeNet3D(nn.Module):
    def __init__(self):
        super(LeNet3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 6, kernel_size=(5, 5, 5))
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



model = LeNet3D()


# Test the model:
x = Variable(torch.randn(1, 3, 32, 32, 32)) # (N,C,D,H,W)
# print(x)
y = model(x)
print(y)



