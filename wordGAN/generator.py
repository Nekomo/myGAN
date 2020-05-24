import torch
import torch.nn as nn
import wordgan_helper
class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    #Sequentialはtorchのモデル定義
    self.fc = nn.Sequential(
      nn.Linear(62, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Linear(1024, 128 * 7 * 7),
      nn.BatchNorm1d(128*7*7),
      nn.ReLU(),
    )
    #deconvolution(逆畳み込み)関数のモデル生成
    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(1024),
      nn.ReLU(),
      nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1),
      nn.Sigmoid(),
    )

    #ネットワーク重みの初期化
    wordgan_helper.initialize_weights(self)

def forward(self, input):
  x = self.conv(input)
  x = x.view(-1, 128 * 7 * 7)
  x.self.fc(x)
  return x