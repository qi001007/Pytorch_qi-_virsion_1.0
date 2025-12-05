"""
数据集自动划分工具（训练集 + 测试集）
================================================
功能：
  将任意「按类别分文件夹」的图片数据集，随机划分为训练集与测试集，
  目录结构保持不变，仅复制文件，不破坏原始数据。

目录要求（输入）：
  data_cat_dog/          # 任意根目录名
  ├─cat/
  │ ├─1.jpg
  │ ├─2.jpg
  │ └─...
  └─dog/
    ├─1.jpg
    └─...

目录输出（自动生成）：
  data/
  ├─train/90% 图像（每类）
  │ ├─cat/
  │ └─dog/
  └─test/10% 图像（每类）
    ├─cat/
    └─dog/

使用方法：
  1. 修改 file_path 变量指向你的原始数据集根目录
  2. 修改 split_rate 变量控制测试集比例（默认 0.1 → 10%）
  3. 运行脚本，等待 processing bar 完成即可

特点：
  - 随机抽样，每次运行结果不同（可固定 random.seed 实现可重复）
  - 仅复制文件，不移动/删除原图，安全快捷
  - 支持任意图片扩展名（需在代码中按需添加）
  - 实时打印进度条，一目了然
"""
import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 获取data文件夹下所有文件夹名（即需要分类的类名）
file_path = 'data_mask'
flower_class = [cla for cla in os.listdir(file_path)]

# 创建 训练集train 文件夹，并由类名在其目录下创建5个子目录
mkfile('data/train')
for cla in flower_class:
    mkfile('data/train/' + cla)

# 创建 验证集val 文件夹，并由类名在其目录下创建子目录
mkfile('data/test')
for cla in flower_class:
    mkfile('data/test/' + cla)

# 划分比例，训练集 : 测试集 = 9 : 1
split_rate = 0.1

# 遍历所有类别的全部图像并按比例分成训练集和验证集
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
    images = os.listdir(cla_path)  # images 列表存储了该目录下所有图像的名称
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
        # eval_index 中保存验证集val的图像名称
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/test/' + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径

        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")
