import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import torch.nn.functional as F

learn_rate = 0.01
tot_iter = 6
batch_size = 4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((255, 255)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataset_train = datasets.ImageFolder('Images/Train', transform=transform)
dataset_test = datasets.ImageFolder('Images/Test', transform=transform)

train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True)
test = torch.utils.data.DataLoader(dataset_test, batch_size=4)

classes = (str(i) for i in range(1, 7))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # ??
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # ??
        self.fc1 = nn.Linear(16*60*60, 120)  # ??
        self.fc2 = nn.Linear(120, 84)  # ??
        self.fc3 = nn.Linear(84, 6)  # ?? num of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 57600)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

n_steps = len(train)
for epoch in range(tot_iter):
    for i, (images, labels) in enumerate(train):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = loss_func(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{tot_iter}], Step [{i+1}/{n_steps}], Loss: {loss.item():.4f}')
            
print('Finished Training')

with torch.no_grad():
    n_corr = 0
    n_samples = 0
    n_class_corr = [0 for i in range(6)]
    n_class_samp = [0 for i in range(6)]
    for images, labels in test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_corr += (predicted == labels).sum().item()

    
    acc = 100 * n_corr/n_samples
    print(f'Accuracy: {acc}%')
