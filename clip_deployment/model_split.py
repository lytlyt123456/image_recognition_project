import torch
import torch.nn as nn
import numpy as np
import model_without_deployment as m

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dict = torch.jit.load('ViT-L-14.pt', map_location='cpu').state_dict()
model = m.build_model(state_dict) # 整个模型

attns = {'visual': [], 'text': []} # 残差块的注意力机制
mlps = {'visual': [], 'text': []} # 残差块的MLP
for i in range(24):
    attn = list(model.visual.transformer.resblocks.children())[i].attn
    attn = attn.to(DEVICE)
    attn.eval()
    attns['visual'].append(attn)

    mlp = list(model.visual.transformer.resblocks.children())[i].mlp
    mlp = mlp.to(DEVICE)
    mlp.eval()
    mlps['visual'].append(mlp)

for i in range(12):
    attn = list(model.transformer.resblocks.children())[i].attn
    attn = attn.to(DEVICE)
    attn.eval()
    attns['text'].append(attn)

    mlp = list(model.transformer.resblocks.children())[i].mlp
    mlp = mlp.to(DEVICE)
    mlp.eval()
    mlps['text'].append(mlp)

text_blocks = model.transformer.resblocks # 文本编码器的残差块集合
visual_blocks = model.visual.transformer.resblocks # 视觉编码器的残差块集合

conv = model.visual.conv1.to(DEVICE) # 视觉编码器的卷积层
conv.eval()

text_proj = model.text_projection.to(DEVICE) # 文本编码器的末端投影层
visual_proj = model.visual.proj # 视觉编码器的末端投影层
