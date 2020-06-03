import os
import pickle
from generator import Generator
from discriminator import Discriminator
#torchvision...ConputerVisionにおける有名データセット・モデルアーキテクチャ・画像変換処理が詰まったライブラリ
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms,utils
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import shutil

import itertools

cuda = torch.cuda.is_available()
if cuda:
  print('cuda available!')
else:
  print('cuda is not available!')

# 　ハイパーパラメータ
batch_size = 1024
lr = 0.0002
z_dim = 62
num_epochs = 100
sample_num = 16
log_dir = './logs'
#画像生成するepoch

image_gen_epoch=[0,1,5,10,20,30,40,50,60,70,80,90,99,100]

def train(D, G, criterion, D_optimizer, G_optimizer, data_loader,epoch):
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

  datacounter = 1
  #データセットをバッチサイズごとに学習
  for batch_idx, (real_images, _) in enumerate(data_loader):
    #ループ数カウンタ
    #print("data_loader_len ,batch_size={},{}", dataloaderlen,batch_size)
    #print("{}/{} trained".format(batch_idx, dataloaderlen))
    if batch_idx*100/dataloaderlen>datacounter:
      print("epoch {} : {}/{} trained({}%)".format(epoch,batch_idx, dataloaderlen,datacounter))
      datacounter+=1
    #データセットの余り，バッチサイズに満たない部分は切り捨て
    if real_images.size()[0] != batch_size:
      break
    
    #乱数
    z=torch.rand((batch_size,z_dim))
    if cuda:
      real_images = real_images.cuda()
      z = z.cuda()

    real_images,z=Variable(real_images),Variable(z)

    #D-①勾配の初期化 TODO:なぜ?
    D_optimizer.zero_grad()

    #D-②discriminatorに本物を渡して識別させる
    #discriminatorの本物画像に識別結果は1に近いほどよい
    D_real = D(real_images)
    #criterion(今回は2値エントロピー)で損失を取る TODO:なぜ2値エントロピー？
    D_real_loss=criterion(D_real,y_real)

    #D-③discriminatorに偽物を渡して識別させる
      #DiscriminatorはGeneratorの識別結果は0に近いほどよい
    #modelに対してinputを渡すと出力が出てくる(この場合乱数z→画像fake_images)
    fake_images = G(z)
    #fake_imagesを通じてGに誤差逆伝播しないようdetachする #TODO:どういうこと？　
    D_fake = D(fake_images.detach())
    #y_fakeはfakeのラベル
    D_fake_loss = criterion(D_fake, y_fake)
    
    #D-④最終的なロスを逆伝播する
    #ロスを足し合わせる
    D_loss = D_real_loss + D_fake_loss
    #微分値の計算
    D_loss.backward()
    #勾配降下(ADAM)によりDのパラメータ更新(Gのパラメータは更新しない)
    D_optimizer.step()
    #epoch全体で出た損失に加算
    D_running_loss += D_loss.data.item()

    #G-①Generator勾配の初期化 TODO;なぜ？
    G_optimizer.zero_grad()

    #G-② Discriminatorに偽物画像を渡して判別させる
    fake_images=G(z)
    D_fake = D(fake_images)
    
    #G-③誤差逆伝播
    #Generatorは，偽物の識別結果が1(本物)に近いほどよい
    #y_realは正解のラベル.偽物画像に対する判断が，本物に近づくように学習する...というワケ
    G_loss = criterion(D_fake, y_real)
    #逆伝播
    G_loss.backward()
    #勾配降下
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
  #driveに退避
  shutil.copy(os.path.join(log_dir, 'epoch_%03d.png' % (epoch)), "../../../drive/\"My Drive\"/gen_results/faceGAN/epoch_%03d.png" % (epoch))
#main

g=Generator()
d = Discriminator()
generator_path="./data/G.pth"
discriminator_path="./data/D.pth"
if os.path.exists(generator_path):
  print("Trained Generator exists.")
  g.load_state_dict(torch.load(generator_path))
if os.path.exists(generator_path):
  print("Trained Discriminator exists.")
  d.load_state_dict(torch.load(discriminator_path))
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

"""
#MNISTデータセットの読み込み
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
"""
# load dataset
transform = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder('data/', transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  
#データ訓練
history = {}
#各epochごとのlossを記録しておく箱
history['D_loss']=[]
history['G_loss']=[]
for epoch in range(num_epochs):
  D_loss, G_loss = train(d, g, criterion, d_optimizer, g_optimizer, data_loader,epoch)
  
  print('epoch %d, D_loss:%.4f G_loss: %.4f' % (epoch + 1, D_loss, G_loss))
  history['D_loss'].append(D_loss)
  history['G_loss'].append(G_loss)

  #特定のエポックでGeneratorから画像を生成
  if epoch in image_gen_epoch:
    #画像を生成
    generate(epoch + 1, g, log_dir)
    #モデルを保存
    torch.save(g.state_dict(),os.path.join(log_dir,'G_%03d.pth'%(epoch+1)))
    torch.save(d.state_dict(), os.path.join(log_dir, 'D_%03d.pth' % (epoch + 1)))
    #モデルを退避
    shutil.copy(os.path.join(log_dir, 'G_%03d.pth' % (epoch + 1)), "../../../drive/\"My Drive\"/gen_results/faceGAN/G_%03d.pth" % (epoch + 1))
    shutil.copy(os.path.join(log_dir, 'D_%03d.pth' % (epoch + 1)), "../../../drive/\"My Drive\"/gen_results/faceGAN/D_%03d.pth" % (epoch + 1))
    



with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
  pickle.dump(history,f)
"""
with open(os.path.join(log_dir, 'history.pkl'), 'rb') as f:
    history = pickle.load(f)
"""
# データの可視化
def imshow(img):
    npimg = img.numpy()
    # [c, h, w] => [h, w, c]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

images, labels = iter(data_loader).next()
images, labels = images[:25], labels[:25]
imshow(utils.make_grid(images, nrow=5, padding=1))
plt.axis('off')

D_loss, G_loss = history['D_loss'], history['G_loss']
plt.plot(D_loss, label='D_loss')
plt.plot(G_loss, label='G_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.savefig('loss.png')