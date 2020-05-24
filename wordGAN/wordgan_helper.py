import torch
import torch.nn as nn
#modelが特定のパターンのときのみ重み初期値を変更
def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.02)
      m.bias.data.zero_()
