import os
import pickle
from generator import Generator
from discriminator import Discriminator
#torchvision...ConputerVisionにおける有名データセット・モデルアーキテクチャ・画像変換処理が詰まったライブラリ
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

cuda = torch.cuda.is_available()
if cuda:
  print('cuda available!')
else:
  print('cuda is not available!')

def train(D, G, criterion, D_optimizer, G_optimizer, data_loader):
  #訓練モードへ
  D.train()
  G.train()

  #本物ラベル = 1
  y_real=Variable(torch.ones(batch_size,1))
  #偽物ラベル = 0
  y_fake = Variable(torch.zeros(batch_size, 1))

  if cuda:
    y_real=y_real.cuda()
    y_fake=y_fake.cuda()
  
  D_running_loss = 0
  G_running_loss = 0

  dataloaderlen=len(data_loader)

  datacounter=1
  for batch_idx, (real_images, _) in enumerate(data_loader):
    if batch_idx>batch_size/1000*datacounter:
      datacounter+=1
      print(datacounter)

    #データセットの余り，バッチサイズに満たない部分は切り捨て
    if real_images.size()[0] != batch_size:
      break
    
    z=torch.rand((batch_size,z_dim))
    if cuda:
      real_images = real_images.cuda()
      z = z.cuda()

    real_images,z=Variable(real_images),Variable(z)


    #①discriminatorの更新
    D_optimizer.zero_grad()
    #discriminatorの本物画像に識別結果は1に近いほどよい
    D_real = D(real_images)
    D_real_loss=criterion(D_real,y_real)

    #DiscriminatorはGeneratorの識別結果は0に近いほどよい
    fake_images=G(z)
    D_fake = D(fake_images.detach())
    D_fake_loss = criterion(D_fake, y_fake)
    
    #最終的なロス
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step() #Gのパラメータを更新しない
    D_running_loss += D_loss.data.item()

    #Generatorの更新
    G_optimizer.zero_grad()

    #Generatorは，偽物の識別結果が1(本物)に近いほどよい
    fake_images=G(z)
    D_fake = D(fake_images)
    G_loss = criterion(D_fake, y_real)
    G_loss.backward()
    G_optimizer.step()
    G_running_loss += G_loss.data.item()
  
  D_running_loss /= len(data_loader)
  G_running_loss /= len(data_loader)

  return D_running_loss,G_running_loss

  #画像生成関数
def generate(epoch, G, log_dir='logs'):
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  
  #潜在変数になる乱数を生成
  sample_z = torch.rand((64, z_dim))
  
  if cuda:
    sample_z = sample_z.cuda()
  sample_z = Variable(sample_z,volatile=True)

  #Generatorでサンプル生成
  samples = G(sample_z).data.cpu()
  save_image(samples,os.path.join(log_dir,'epoch_%03d.png' % (epoch)))



#main


# 　ハイパーパラメータ
batch_size = 128
lr = 0.0002
z_dim = 62
num_epochs = 25
sample_num = 16
log_dir = './logs'

g=Generator()
d = Discriminator()
print(g)
print(d)

#cudaがあるなら高速化可能
if cuda:
  g.cuda()
  d.cuda()

#optimizer
g_optimizer = optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
#loss
criterion = nn.BCELoss()

#MNISTデータセットの読み込み()
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#データ訓練
history = {}
history['D_loss']=[]
history['G_loss']=[]
image_gen_epoch=[0, 9, 24]
for epoch in range(num_epochs):
  print(epoch)
  D_loss, G_loss = train(d, g, criterion, d_optimizer, g_optimizer, data_loader)
  
  print('epoch %d, D_loss:%.4f G_loss: %.4f' % (epoch + 1, D_loss, G_loss))
  history['D_loss'].append(D_loss)
  history['G_loss'].append(G_loss)

  #特定のエポックでGeneratorから画像を生成
  if epoch in image_gen_epoch:
    generate(epoch + 1, g, log_dir)
    #モデルも保存
    torch.save(g.state_dict()),os.path.join(log_dir,'G_%03d.pth'%(epoch+1))
    torch.save(d.state_dict()),os.path.join(log_dir,'D_%03d.pth'%(epoch+1))

with open(os.path, join(log_dir, 'history.pkl'), 'wb') as f:
  pickle.dump(history,f)