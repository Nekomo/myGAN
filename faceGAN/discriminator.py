import torch
import torch.nn as nn
import wordgan_helper

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    """
    self.conv = nn.Sequential(
      #maxpoolingを用いないのもDCGANの特徴らしい
      #畳み込み
      nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
      #LeakyReLUを用いるのはDCGANの特徴
      nn.LeakyReLU(0.2),
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
      #バッチノーマライゼーション(標準化)
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2)
    )
    self.fc = nn.Sequential(
      nn.Linear(128 * 7 * 7, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 1),
      #1次元にした特徴をsigmoidで判別
      nn.Sigmoid(),
    )
    """
    self.conv = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2),
    )
    
    self.fc = nn.Sequential(
      nn.Linear(128 * 16 * 16, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 1),
      nn.Sigmoid(),
    )

    wordgan_helper.initialize_weights(self)

  def forward(self, input):
    x = self.conv(input)
    #7*7の画像128枚を1次元にする(flatten)
    x = x.view(-1, 128 * 16 * 16)
    #128*7*7→スカラーにしてsigmoidに通す
    x = self.fc(x)
    return x
