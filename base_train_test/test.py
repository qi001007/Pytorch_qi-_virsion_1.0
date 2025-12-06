import torch
import json
import yaml
import argparse
import importlib

from pathlib import Path
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import numpy as np
import torch.utils.data as Data

from tools.my_bar import simple_bar


# 创建解析器
parser = argparse.ArgumentParser()
# 读取 --config 参数
parser.add_argument('--config', default='config.yaml')
# 读取 --process 参数
parser.add_argument('--process', default='process.yaml')
# 读取 --mean_std_stats 参数
parser.add_argument('--mean_std_stats')
# 读取 --best_wst_path 参数
parser.add_argument('--best_wst_path')
# 开始解析
args = parser.parse_args()

# 获取config.yaml存为py的字典
with (open(args.config, mode='r', encoding='utf-8') as c,
      open(args.process, mode='r', encoding='utf-8') as p,
      open(args.mean_std_stats, mode='r', encoding='utf-8') as ms,
      open(args.best_wst_path, mode='r', encoding='utf-8') as wt):
    config = yaml.safe_load(c)
    process = yaml.safe_load(p)
    mean_std = json.load(ms)
    best_wst = json.load(wt)
# 构造新字典，将每一部分的名字和内部参数对应，方便调运
global_params = {part['name']: part for part in config['global']}
# 构造新字典，将每一步step_name和内部参数对应方便调运
pipeline = {step['name']: step for step in process['process']}
# 构造输出路径，存储变量
out_path = Path(process['project_base']['project_path']) / pipeline['test']['output']
# 自动递归建目录文件
out_path.parent.mkdir(parents=True, exist_ok=True)

# 从config的global中获取model_path，model_name
model_path = global_params['model']['model_path']
model_name = global_params['model']['model_name']
# 动态导入模块
Module = importlib.import_module(model_path)   # 等价于 import models
# 从模块里取出类
module = getattr(Module, model_name)

# 取出参数
size = tuple(global_params['img']['img_size'])
batch_size = int(global_params['test']['batch_size'])
num_work = int(global_params['test']['num_work'])
test_way = int(global_params['test']['test_way'])


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size), transforms.ToTensor()]),
                             download=True)

    class_label = test_data.classes  # 标签

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_work)

    return test_dataloader, class_label


def test_model_process(model, test_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # 指定训练设备
    model = model.to(device)        # 将模型放入设备

    # 初始化参数
    # model测试精度
    test_corrects = 0.0
    test_num = 0.0

    # 只进行前向传播计算，不计算梯度，关闭梯度计算，加快运行速度
    with torch.no_grad():
        # 让 tqdm 包裹 dataloader，出现进度条
        for step, (test_data_x, test_data_y) in simple_bar(test_dataloader, desc="Testing", unit="img"):
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data).item()
            # 当前用于预测的样本数量
            test_num += test_data_x.size(0)

    # 计算准确率
    test_acc = np.double(test_corrects) / test_num
    print(f"测试准确率：{test_acc:.4f}")


def test_model_process_show(model, test_dataloader, class_label):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # 指定训练设备
    model = model.to(device)        # 将模型放入设备

    # 初始化参数
    # model测试精度
    test_corrects = 0.0
    test_num = 0.0

    # 只进行前向传播计算，不计算梯度，关闭梯度计算，加快运行速度
    with torch.no_grad():
        # 让 tqdm 包裹 dataloader，出现进度条
        for step, (test_data_x, test_data_y) in enumerate(test_dataloader):
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data).item()
            # 当前用于预测的样本数量
            test_num += test_data_x.size(0)
            # 展现预测值和标签
            result = pre_lab.item()
            label = test_data_y.item()
            print("预测值：", class_label[result], "------------", "真实值：", class_label[label])

    # 计算准确率
    test_acc = np.double(test_corrects) / test_num
    print(f"测试准确率：{test_acc:.4f}")


if __name__ == '__main__':
    # 加载模型
    Model = module()
    # 存入参数
    Model.load_state_dict(torch.load(best_wst))
    # 加载测试数据
    Test_dataloader, Class_label = test_data_process()
    # 开始测试
    if test_way == 1:
        test_model_process(Model, test_dataloader=Test_dataloader)
    elif test_way == 2:
        test_model_process_show(Model, test_dataloader=Test_dataloader, class_label=Class_label)
    else:
        print('请重新尝试，test_way = 1（只显示准确率）或2（显示准确率和文本）')
