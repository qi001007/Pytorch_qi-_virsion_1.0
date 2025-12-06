"""
数据集 RGB 均值与方差计算器
====================================
功能：
  递归扫描指定文件夹下所有图片（jpg/png/bmp...），
  采用「两遍扫描法」计算整个数据集的 RGB 均值（mean）与方差（variance），
  像素值先归一化到 [0, 1] 区间，结果可直接用于 PyTorch transforms.Normalize()。

算法步骤：
  1. 第一遍：累加所有像素的通道和 S，并统计总像素数 N
     μ = S / N
  2. 第二遍：累加平方差 Q = Σ(x - μ)²
     σ² = Q / N
  时间复杂度 O(N)，空间复杂度 O(1)。

用法：
  修改 folder_path 后运行即可；打印出的 mean 与 variance
  复制到 transforms.Normalize(mean, std) 即可使用。

注意：
  - 灰度图或通道数不符的图片会被跳过（ValueError 捕获）
  - 大数据集耗时较长，建议 SSD 并行读取（可扩展用 Pillow-SIMD、tqdm）
"""
import os
import json
import yaml
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from my_bar import simple_bar


# 创建解析器
parser = argparse.ArgumentParser()
# 读取 --config 参数
parser.add_argument('--config', default='config.yaml')
# 读取 --process 参数
parser.add_argument('--process', default='process.yaml')
# 开始解析
args = parser.parse_args()

# 获取config.yaml存为py的字典
with open(args.config, mode='r', encoding='utf-8') as c, open(args.process, mode='r', encoding='utf-8') as p:
    config = yaml.safe_load(c)
    process = yaml.safe_load(p)
# 构造新字典，将每一部分的名字和内部参数对应，方便调运
global_params = {part['name']: part for part in config['global']}
# 构造新字典，将每一步step_name和内部参数对应方便调运
pipeline = {step['name']: step for step in process['process']}
# 构造输出路径，存储变量
out_path = Path(process['project_base']['project_path']) / pipeline['mean_std']['output']
# 自动递归建目录文件
out_path.parent.mkdir(parents=True, exist_ok=True)

# 文件夹路径，包含所有图片文件
folder_path = global_params['data_file']['data_file_path']

# 初始化累积变量
total_pixels = 0
sum_normalized_pixel_values = np.zeros(3)  # 如果是RGB图像，需要三个通道的均值和方差

# 遍历文件夹中的图片文件
for root, dirs, files in os.walk(folder_path):
    for idx, filename in simple_bar(files, desc=f"{root[-30:]}_mean_running"):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 可根据实际情况添加其他格式
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)

            # 归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0

            # print(image_path)
            # print(normalized_image_array.shape)
            # 累积归一化后的像素值和像素数量
            total_pixels += normalized_image_array.size
            sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

# 计算均值和方差
mean = sum_normalized_pixel_values / total_pixels


sum_squared_diff = np.zeros(3)
for root, dirs, files in os.walk(folder_path):
    for idx, filename in simple_bar(files, desc=f"{root[-30:]}_variance_running"):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            # 归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0
            # print(normalized_image_array.shape)
            # print(mean.shape)
            # print(image_path)

            try:
                diff = (normalized_image_array - mean) ** 2
                sum_squared_diff += np.sum(diff, axis=(0, 1))
            except ValueError:
                print(f"捕获到自定义异常")
            # diff = (normalized_image_array - mean) ** 2
            # sum_squared_diff += np.sum(diff, axis=(0, 1))

variance = sum_squared_diff / total_pixels

print("Mean:", mean.tolist())
print("Variance:", variance.tolist())

mean_std = {"Mean": mean.tolist(),
            "Variance": variance.tolist()}
json.dump(mean_std, open(out_path, mode='w', encoding='utf-8'), indent=2)
