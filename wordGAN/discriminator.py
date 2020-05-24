import torch
import torch.nn as nn
import wordgan_helper

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2)
    )
    self.fc = nn.Sequential(
      nn.Linear(128 * 7 * 7, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 1),
      nn.Sigmoid(),
    )

    wordgan_helper.initialize_weights(self)

def forward(self, input):
  x = self.conv(input)
  x = view(-1, 128 * 7 * 7)
  x = self.fc(x)
  return x