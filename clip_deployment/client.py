"""
本部分为后端调用接口。

1. build_model_and_deploy 方法用于模型创建与部署，传入的参数决定模型部署方式。具体如下：

(1) complete_encoders_requires_server: bool
整个图像编码器和文本编码器编码图像和文本的过程交给服务器进行处理，在条件允许的情况下引入并行
当 complete_encoders_requires_server == True 时，后面的参数除 cos_sim 外可任意设置

(2) resblocks_requires_server: bool,
图像编码器和文本编码器的所有残差块部署到服务器，不包括首部嵌入层和末端投影层
当 resblocks_requires_server = True 时，attn_requires_server 和 mlp_requires_server 可任意设置

(3) vision_conv_requires_server: bool,
图像编码器的嵌入层（卷积层）被部署到服务器

(4) attn_requires_server: bool,
将多头自注意力部分部署到服务器

(5) cos_sim_requires_server: bool,
将最终求图像-文本相似度的过程交给服务器进行处理

(6) mlp_requires_server: bool,
将每个残差块的MLP部分部署到服务器

(7) proj_requires_server: bool
将图像和文本编码器末端的投影层部署到服务器

2. predict 方法用于预测图像类别。

传入参数如下：

(1) model 为创建并部署好的模型
(2) image_paths: List[str] 为所有图像路径的列表
(3) class_names: List[str] 为所有图像类别名称的列表
"""

import torch
from torchvision import transforms
import model as m
from typing import List
import clip_
import cv2

state_dict = torch.jit.load('ViT-L-14.pt', map_location='cpu').state_dict()
transform = transforms.ToTensor()

def build_model_and_deploy(
        complete_encoders_requires_server: bool,
        resblocks_requires_server: bool,
        vision_conv_requires_server: bool,
        attn_requires_server: bool,
        cos_sim_requires_server: bool,
        mlp_requires_server: bool,
        proj_requires_server: bool
):
    model = m.build_model(
        state_dict,
        complete_encoders_requires_server,
        resblocks_requires_server,
        vision_conv_requires_server,
        attn_requires_server,
        cos_sim_requires_server,
        mlp_requires_server,
        proj_requires_server
    )

    return model

def predict(model, image_paths: List[str], class_names: List[str]):
    prompts = ['a photo of a ' + class_name for class_name in class_names]
    tokenized_prompts = clip_.tokenize(prompts)

    imgs = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)  # [C, H, W]
        imgs.append(img)
    imgs = torch.stack(imgs)  # [B, C, H, W]

    with torch.no_grad():
        logits_per_image, logits_per_text = model(imgs, tokenized_prompts)

    return logits_per_image, logits_per_text


# test
# if __name__ == '__main__':
#     model = build_model_and_deploy(False, False,
#                                    False, False,
#                                    False, False, True)
#     logits_per_image, _ = predict(model, ['C:/Users/yanta/Desktop/image_0005.jpg'],
#                                   ['dog', 'cat', 'cup', 'bowl', 'water'])
#     print(logits_per_image)

