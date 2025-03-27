import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
    
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(
        out_channels, out_channels,
        kernel_size=3, stride=1, padding=1
    )
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
            )
    self.relu = nn.ReLU()

  def forward(self, x):
    shortcut = x.clone()
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x += self.downsample(shortcut)
    x = self.relu(x)
    return x
  
class ResNet(nn.Module):
  def __init__(self, n_classes):
      super().__init__()
      self.model = nn.Sequential(
          ConvBlock(3, 64),
          ConvBlock(64, 64),
          ResidualBlock(64, 64),
          ResidualBlock(64, 64),

          ConvBlock(64, 128),
          ConvBlock(128, 128),
          ResidualBlock(128, 128),
          ResidualBlock(128, 128),

          ConvBlock(128, 256),
          ConvBlock(256, 256),
          ResidualBlock(256, 256),
          ResidualBlock(256, 256),

          nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Linear(256, n_classes)

  def forward(self, x):
      x = self.model(x)
      x = torch.flatten(x, 1)
      return self.fc(x)
