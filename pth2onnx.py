import torch.onnx
# import onnxruntime as ort
from model import U2NETP

# 创建.pth模型

model = U2NETP(3, 1)
# 加载权重
model_path = 'saved_models/u2netp/u2netp_bce_itr_180000_train_0.044270_tar_0.003919.pth'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model_statedict = torch.load(model_path, map_location=device)
model.load_state_dict(model_statedict)

model.to(device)
model.eval()

input_data = torch.randn(1, 3, 320, 320, device=device)

# 转化为onnx模型
input_names = ['input']
output_names = ['output']

torch.onnx.export(model, input_data, 'model.onnx', opset_version=11, verbose=True, input_names=input_names,
                  output_names=output_names)