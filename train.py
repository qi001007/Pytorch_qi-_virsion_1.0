import copy
import os
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.amp import autocast, GradScaler
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import pandas as pd
from pynput import keyboard

from my_bar import simple_bar


# -------------------- 训练函数 --------------------
def train_val_data_process():
    data_ = FashionMNIST(root='./data',
                         train=True,
                         transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                         download=True)
    train_data, val_data = Data.random_split(data_,
                                             lengths=[round(0.8*len(data_)), round(0.2*len(data_))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=4)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=64,
                                     shuffle=True,
                                     num_workers=4)
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # 指定训练设备
    print(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)      # 创建优化器（梯度下降法的优化方法：SGD,Adam,A,B,C）
    criterion = nn.CrossEntropyLoss()       # 创建损失函数（均方误差损失函数：回归,交叉熵损失函数Cross-Entropy Loss：分类）
    model = model.to(device)        # 将模型放到训练设备当中
    best_model_wts = copy.deepcopy(model.state_dict())      # 复制当前模型参数
    scaler = GradScaler()

    # 初始化参数
    # 最高精确度
    best_acc = 0.0
    # 训练集损失函数列表
    train_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 最终信息表
    train_process = pd.DataFrame(columns=["epoch", "train_loss_all", "train_acc_all", "val_loss_all", "val_acc_all"])
    # 当前时间
    since = time.time()
    date_str_start = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    os.makedirs(f'./model_wts/{date_str_start}model', exist_ok=True)

    for i, epoch in simple_bar(num_epochs, flag=kill_epoch, is_open_bar=False):
        print("-"*50)
        print(f"Epoch {epoch+1}/{num_epochs}")

        # 初始化每一轮的参数
        # 训练集损失值
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0.0
        # 验证集损失值
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0.0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        # 打开梯度计算
        with torch.enable_grad():
            for step, (b_x, b_y) in simple_bar(train_dataloader, flag=kill_epoch,
                                               desc="Train_dataloader_running", step_size=2):
                if kill_epoch:
                    print("\n[q] 被按下，立即终止训练。")
                    # raise KeyboardInterrupt
                    break
                # 将特征放入到训练设备中
                b_x = b_x.to(device)
                # 将标签放入到训练设备中
                b_y = b_y.to(device)
                # 设置模型为训练模式
                model.train()

                # 把原来的 forward + backward + step 三步换成 “autocast 包前向 + scaler 包反向”
                # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                # 前向：只在 autocast 上下文里做
                with autocast(str(device)):
                    output = model(b_x)
                    # 查找每一行中最大值对应的行标
                    pre_lab = torch.argmax(output, dim=1)
                    # 计算每一个batch的损失值
                    loss = criterion(output, b_y)

                # 将梯度初始化为0
                optimizer.zero_grad()
                # 反向：先放大损失，再反向
                scaler.scale(loss).backward()
                # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
                # 更新：scaler负责把梯度缩回去并step
                scaler.step(optimizer)
                # 更新缩放因子
                scaler.update()
                # 对损失函数进行累加
                train_loss += loss.item() * b_x.size(0)
                # 如果预测正确，则准确度train_corrects加1
                train_corrects += torch.sum(pre_lab == b_y.data).item()
                # 当前用于训练的样本数量
                train_num += b_x.size(0)

        # 关闭梯度计算
        with torch.no_grad():
            for step, (b_x, b_y) in simple_bar(val_dataloader, flag=kill_epoch,
                                               desc="Val_dataloader_running", step_size=2):
                if kill_epoch:
                    print("\n[q] 被按下，立即终止训练。")
                    # raise KeyboardInterrupt
                    break
                # 将特征放入到验证设备中
                b_x = b_x.to(device)
                # 将标签放入到验证设备中
                b_y = b_y.to(device)
                # 设置模型为评估模式
                model.eval()

                # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                # 前向：用 autocast 跑 FP16
                with autocast(str(device)):
                    output = model(b_x)
                    # 查找每一行中最大值对应的行标
                    pre_lab = torch.argmax(output, dim=1)
                    # 计算每一个batch的损失值
                    loss = criterion(output, b_y)
                # 对损失函数进行累加
                val_loss += loss.item() * b_x.size(0)
                # 如果预测正确，则准确度train_corrects加1
                val_corrects += torch.sum(pre_lab == b_y.data).item()
                # 当前用于验证的样本数量
                val_num += b_x.size(0)

        # 跳出条件
        if kill_epoch:
            print("\n[q] 被按下，立即终止训练。")
            # raise KeyboardInterrupt
            break

        # 计算并保存每一次迭代的损失值和准确率
        if train_num > 0:
            # 计算并保存训练集的loss值
            train_loss_all.append(train_loss / train_num)
            # 计算并保存训练集的准确率
            train_acc_all.append(np.double(train_corrects) / train_num)
        if val_num > 0:
            # 计算并保存验证集的loss值
            val_loss_all.append(val_loss / val_num)
            # 计算并保存验证集的准确率
            val_acc_all.append(np.double(val_corrects) / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        # [-1] 是 Python 列表索引语法，表示 “最后一个元素”
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重
        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 每轮训练的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

        # 选择每轮最优参数
        if epoch + 1 == num_epochs:
            # 如果训练执行到最后循环，将最后的最优参数保存，删掉best_model_epoch_break
            date_str_end = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
            torch.save(best_model_wts,
                       f'./model_wts/'
                       f'{date_str_start}model/best_model_{date_str_end}.pth')
            last_ckpt = (f'./model_wts/'
                         f'{date_str_start}model/best_model_{num_epochs - 1}epoch.pth')
            if os.path.exists(last_ckpt):
                os.remove(last_ckpt)
        else:
            torch.save(best_model_wts,
                       f'./model_wts/'
                       f'{date_str_start}model/best_model_{epoch + 1}epoch.pth')
            old_ckpt = (f'./model_wts/'
                        f'{date_str_start}model/best_model_{epoch}epoch.pth')
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

        # 将关键数据存入pandas的DtaFrame中，方便之后画图调用
        train_process = pd.DataFrame({"epoch": range(len(train_loss_all)),
                                      "train_loss_all": train_loss_all,
                                      "train_acc_all": train_acc_all,
                                      "val_loss_all": val_loss_all[:len(train_loss_all)],  # 保证长度一致
                                      "val_acc_all": val_acc_all[:len(train_loss_all)]})

    return train_process, date_str_start


def matplot_acc_loss(train_process, date_str_start):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_all"], 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc_all"], 'bs-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.savefig(f'./model_wts/{date_str_start}model/loss_acc.jpg')
    plt.show()


# -------------------- 热键退出 --------------------
def on_press(key):
    global kill_epoch
    try:
        if key.char and key.char.lower() == 'q':
            # 只改标志，不 raise
            kill_epoch = True
            # 给自己发 SIGINT，等价于 Ctrl-C
            # signal.raise_signal(signal.SIGINT)
    except AttributeError:
        pass


def _gentle_shutdown(loader):
    """温和关闭 DataLoader 的多进程 worker，不抛异常、不报警告。"""
    itr = getattr(loader, '_iterator', None)          # 拿到迭代器
    if itr:
        shutdown = getattr(itr, '_shutdown_workers', None)
        if callable(shutdown):                        # 如果方法存在就调
            shutdown()


# -------------------- Windows 安全入口 --------------------
def main():
    global kill_epoch
    # 退出标志位
    kill_epoch = False
    # 启动后台监听线程（非阻塞）
    listener = keyboard.Listener(on_press=on_press, daemon=True)
    listener.start()
    # 将模型实例化
    Model = LeNet()
    # 数据分批
    Train_dataloader, Val_dataloader = train_val_data_process()
    # 参数初始化
    Train_process = pd.DataFrame(
        columns=["epoch", "train_loss_all", "train_acc_all", "val_loss_all", "val_acc_all"])
    Date_str_start = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    try:
        # 开始训练
        Train_process, Date_str_start = train_model_process(Model, Train_dataloader, Val_dataloader, num_epochs=50)
        # 绘制图形
        matplot_acc_loss(Train_process, Date_str_start)
    except KeyboardInterrupt:
        print("\n[停止] 被按下，已停止训练。")
        # 绘制图形
        matplot_acc_loss(Train_process, Date_str_start)
        print(Train_process, Date_str_start)
    finally:
        _gentle_shutdown(Train_dataloader)
        _gentle_shutdown(Val_dataloader)
        listener.stop()


if __name__ == "__main__":
    kill_epoch = False
    main()
