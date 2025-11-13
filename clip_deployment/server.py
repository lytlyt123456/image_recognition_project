from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import io
import base64
import model_without_deployment as m

app = Flask(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载视觉编码器和文本编码器的所有多头自注意力模块
state_dict = torch.jit.load('ViT-L-14.pt', map_location='cpu').state_dict()
model = m.build_model(state_dict)

# 残差块的注意力机制与MLP
attns = {'visual': [], 'text': []}
mlps = {'visual': [], 'text': []}
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

# 视觉编码器的卷积层
conv = model.visual.conv1.to(DEVICE)
conv.eval()

# 文本编码器和视觉编码器的末端投影层
text_proj = model.text_projection.to(DEVICE)
visual_proj = model.visual.proj

# 文本和编码器的encoder blocks
text_blocks = model.transformer.resblocks
visual_blocks = model.visual.transformer.resblocks

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'visual_blocks': len(attns['visual']),
        'text_blocks': len(attns['text'])
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'MHSA Server is running'})

@app.route('/attention', methods=['POST'])
def attention():
    # 接收base64编码的数据
    data = request.json
    data = base64.b64decode(data['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE) # 输入的张量
    encoder_type = tensor_dict['encoder_type'] # 文本编码器 or 视觉编码器
    block_num = tensor_dict['block_num'] # 第几个残差块
    attn = attns[encoder_type][block_num] # 提取对应的attention模块
    attn_mask = tensor_dict['attn_mask']
    if attn_mask is not None:
        attn_mask = attn_mask.to(DEVICE)

    # 推理
    with torch.no_grad():
        output = attn(query=x, key=x, value=x, need_weights=False, attn_mask=attn_mask)[0]

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/cos_sim', methods=['POST'])
def cos_sim():
    # 接收base64编码的数据
    data = request.json
    data = base64.b64decode(data['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')
    image_features = tensor_dict['image_features'].to(DEVICE)
    text_features = tensor_dict['text_features'].to(DEVICE)
    logit_scale = tensor_dict['logit_scale'].to(DEVICE)

    logits_per_image = logit_scale * image_features @ text_features.t()

    buffer = io.BytesIO()
    torch.save({'output': logits_per_image.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/mlp', methods=['POST'])
def mlp():
    # 接收base64编码的数据
    data = request.json
    data = base64.b64decode(data['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量
    encoder_type = tensor_dict['encoder_type']  # 文本编码器 or 视觉编码器
    block_num = tensor_dict['block_num']  # 第几个残差块
    mlp = mlps[encoder_type][block_num]  # 提取对应的mlp模块

    # 推理
    with torch.no_grad():
        output = mlp(x)

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/vision_conv', methods=['POST'])
def vision_conv():
    # 接收base64编码的数据
    data = request.json
    data = base64.b64decode(data['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量

    # 推理
    with torch.no_grad():
        output = conv(x)

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/text_projection', methods=['POST'])
def text_projection():
    # 接收base64编码的数据
    data = request.json
    data = base64.b64decode(data['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量

    # 推理
    with torch.no_grad():
        output = x @ text_proj

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/visual_projection', methods=['POST'])
def visual_projection():
    # 接收base64编码的数据
    data = request.json
    data = base64.b64decode(data['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量

    # 推理
    with torch.no_grad():
        output = x @ visual_proj

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/encoder_blocks', methods=['POST'])
def encoder_blocks():
    # 接收base64编码的数据
    data = request.json
    data = base64.b64decode(data['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量
    encoder_type = tensor_dict['encoder_type']  # 文本编码器 or 视觉编码器

    if encoder_type == 'text':
        blocks = text_blocks
    else:
        blocks = visual_blocks

    # 推理
    with torch.no_grad():
        output = blocks(x)

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/complete_encoders', methods=['POST'])
def complete_encoders():
    # 接收base64编码的数据
    data = request.json
    data = base64.b64decode(data['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    image = tensor_dict['image'].to(DEVICE)  # 输入的张量
    text = tensor_dict['text'].to(DEVICE)  # 文本编码器 or 视觉编码器

    # 推理
    with torch.no_grad():
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:  # 并行执行
            inputs = [(image,), (text,)]
            models = [model.encode_image, model.encode_text]
            outputs = nn.parallel.parallel_apply(models, inputs)
            image_features, text_features = outputs
        else:
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({
        'image_features': image_features.cpu(),
        'text_features': text_features.cpu()
    }, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
