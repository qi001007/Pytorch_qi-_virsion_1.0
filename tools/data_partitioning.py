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
import yaml
import shutil
import random
import argparse
from pathlib import Path


# 创建解析器
parser = argparse.ArgumentParser()
# 读取 --config 参数
parser.add_argument('--config', default='config.yaml')
# 开始解析
args = parser.parse_args()

# 获取config.yaml存为py的字典
with open(args.config, mode='r', encoding='utf-8') as c:
    config = yaml.safe_load(c)
# 构造新字典，将每一部分的名字和内部参数对应，方便调运
global_params = {part['name']: part for part in config['global']}
# 获取data文件夹下所有文件夹名（即需要分类的类名）
file_path = Path(global_params['data_file']['data_file_path'])
# 划分比例，训练集 : 测试集 = 9 : 1
split_rate = global_params['data_file']['split_rate']


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 列出所有“目录”当作类别
flower_class = [c for c in file_path.iterdir() if c.is_dir()]

# 创建 训练集train 文件夹，并由类名在其目录下创建5个子目录
# 创建 验证集val 文件夹，并由类名在其目录下创建子目录
for split in ('train', 'test'):
    for cls in flower_class:
        (Path('data') / split / cls.name).mkdir(parents=True, exist_ok=True)

# 遍历所有类别的全部图像并按比例分成训练集和验证集
for cls_dir in flower_class:
    images = list(cls_dir.iterdir())
    num = len(images)
    val_imgs = set(random.sample(images, k=int(num * split_rate)))

    for idx, img in enumerate(images):
        split_folder = 'test' if img in val_imgs else 'train'
        dst = Path('data') / split_folder / cls_dir.name / img.name
        shutil.copy(str(img), str(dst))   # pathlib→str
        print(f"\r[{cls_dir.name}] processing [{idx+1}/{num}]", end="")
    print()

print("processing done!")
