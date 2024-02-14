import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from hyperparameter import *


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.avgpool1(x)
        x = F.tanh(self.conv2(x))
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class ResNet50Like(nn.Module):
    def __init__(self, num_classes=62):
        super(ResNet50Like, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


class AlexNet(nn.Module):
    def __init__(self, num_classes=62):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Adjusted kernel size and stride
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Adjusted kernel size and stride
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Adjusted kernel size and stride
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Adjusted kernel size and stride
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # Use num_classes parameter
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return F.softmax(x, dim=1)
    


# GoogleLeNet
# class GoogLeNet(nn.Module):
#     def __init__(self, num_classes=62):
#         super(GoogLeNet, self).__init__()

#         # First convolutional layer
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # Second convolutional layer
#         self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # Inception modules
#         self.inception3a = InceptionModule(192, 64, 128, 128, 32, 32, 32)
#         self.inception3b = InceptionModule(256, 128, 192, 192, 96, 96, 64)

#         self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)

#         self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

#         # Global average pooling
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         # Fully connected layer
#         self.fc = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.maxpool1(x)

#         x = F.relu(self.conv2(x))
#         x = self.maxpool2(x)

#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.maxpool3(x)

#         x = self.inception4a(x)
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)
#         x = self.inception4e(x)
#         x = self.maxpool4(x)

#         x = self.inception5a(x)
#         x = self.inception5b(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return F.softmax(x, dim=1)



class ResNeXT50(nn.Module):
    def __init__(self, num_classes=62):
        super(ResNeXT50, self).__init__()
        self.resnetxt50 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

        # Anpassung der ersten Conv2d-Layer für einen Eingangskanal
        self.resnetxt50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Aktualisierung der Eingabe-Features für die Fully Connected-Layer
        in_features = self.resnetxt50.fc.in_features
        self.resnetxt50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnetxt50(x)
    
