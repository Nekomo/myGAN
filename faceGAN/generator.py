import torch
import torch.nn as nn
import wordgan_helper
class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    """
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
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1),
      nn.Sigmoid(),
    )
    """
    self.fc = nn.Sequential(
      nn.Linear(62, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Linear(1024, 128 * 16 * 16),
      nn.BatchNorm1d(128 * 16 * 16),
      nn.ReLU(),
    )

    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
      nn.Sigmoid(),
    )

    #ネットワーク重みの初期化
    wordgan_helper.initialize_weights(self)

  def forward(self, input):
    #62次元の乱数を128*7*7次元のベクトルにする
    x = self.fc(input)
    """
    #1次元のxを7*7の特徴画像128個に展開
    x = x.view(-1, 128, 7, 7)
    """
    x = x.view(-1, 128, 16, 16)
    #逆畳み込み
    x=self.deconv(x)
    return x
