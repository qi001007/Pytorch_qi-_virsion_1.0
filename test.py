import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet
import numpy as np
from my_bar import simple_bar


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                             download=True)

    class_label = test_data.classes  # 标签

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=4)

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
    Model = LeNet()
    # 存入参数
    Model.load_state_dict(torch.load("./model_wts/"
                                     "2025-11-20-22-49model/best_model_2025-11-20-23-05.pth"))
    # 加载测试数据
    Test_dataloader, Class_label = test_data_process()
    # 开始测试
    # test_model_process(Model, test_dataloader=Test_dataloader)
    test_model_process_show(Model, test_dataloader=Test_dataloader, class_label=Class_label)
