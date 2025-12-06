import os
import cv2
import json
import yaml
import time
import torch
import argparse
import importlib

from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np

from tools.my_bar import simple_bar

# 创建解析器
# formatter_class=argparse.ArgumentDefaultsHelpFormatter 是 argparse 提供的一种“帮助文本格式化器”。
# 作用：让 python xxx.py -h 打印帮助信息时，自动把各参数的“默认值”列出来，用户不用翻代码就能知道缺省是多少
parser = argparse.ArgumentParser(
    description="PyTorch Qi版框架 - 训练入口",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# 读取 --config 参数
parser.add_argument('--config', type=str, required=True, default='config.yaml')
# 读取 --process 参数
parser.add_argument('--process', type=str, required=True, default='process.yaml')
# 读取 --mean_std_stats 参数
parser.add_argument('--mean_std_stats', type=str, required=True, default='Intermediate_data/mean_std.json')
# 读取 --best_wst_path 参数
parser.add_argument('--best_wst_path', type=str, required=True, default='Intermediate_data/best_wst_path.json')
# 读取 --test_data_dir 参数
parser.add_argument('--test_data_dir', type=str, required=True, default='data/test')
# 读取 --train_data_dir 参数
parser.add_argument('--train_data_dir', type=str, required=True, default='data/train')

# 开始解析
args = parser.parse_args()
# print("argparse 通过！", args)

# 获取config.yaml以及其他yaml和json文件内容存为py的字典
with (open(args.config, mode='r', encoding='utf-8') as c,
      open(args.process, mode='r', encoding='utf-8') as p,
      open(args.mean_std_stats, mode='r', encoding='utf-8') as ms,
      open(args.best_wst_path, mode='r', encoding='utf-8') as wt,
      ):
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

# 从config的global中获取model_path，model_name，model_part_name
model_path = global_params['model']['model_path']
model_name = global_params['model']['model_name']
model_part_name = global_params['model']['model_part_name']
# 动态导入模块
Module = importlib.import_module(model_path)   # 等价于 import models
# 从模块里取出类
module = getattr(Module, model_name)
module_part = getattr(Module, model_part_name)

# 取出参数
size = tuple(global_params['img']['img_size'])

mean = np.array(mean_std['Mean'])
std = np.array(mean_std['Variance'])

img_channels = global_params['img']['img_channels']
out_channels = global_params['img']['out_channels']

Test_data_root = args.test_data_dir
Train_data_root = args.train_data_dir


class InferenceConfig:
    """配置类，便于替换模型、数据集和数据处理方法"""
    # 模型配置
    MODEL_CLASS = module
    MODEL_ARGS = (module_part,)  # 模型初始化参数
    MODEL_WEIGHTS_PATH = best_wst

    # 数据预处理配置
    IMAGE_SIZE = size
    MEAN = mean
    STD = std

    # UI配置
    UI_HEIGHT = 100
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 700
    KEY_REPEAT_DELAY = 0.08  # 更快地翻页速度

    # 图像扩展名
    IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff')

    @classmethod
    def get_transform(cls):
        """获取数据预处理流程"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cls.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(cls.MEAN, cls.STD)
        ])


class InferenceUI:
    def __init__(self, test_data_root, train_data_root, in_config=InferenceConfig):
        self.config = in_config

        # 获取类别名称
        train_set = ImageFolder(root=train_data_root, transform=transforms.ToTensor())
        self.class_names = train_set.classes
        print('=> 训练集类别顺序:', self.class_names)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._load_model()

        # 获取图像列表
        self.img_list = self._get_image_list(test_data_root)
        assert self.img_list, f'No image found in {test_data_root}'
        print(f'=> 共找到 {len(self.img_list)} 张测试图')

        # 数据预处理
        self.transform = self.config.get_transform()

        # 状态变量
        self.cache = {}
        self.global_done = False
        self.idx = 0
        self.last_key_time = 0

        # 初始化UI
        self._init_ui()

    def _load_model(self):
        """加载模型 - 可根据需要重写此方法"""
        model = self.config.MODEL_CLASS(*self.config.MODEL_ARGS, img_channels=img_channels, out_channels=out_channels)
        model.load_state_dict(torch.load(self.config.MODEL_WEIGHTS_PATH, map_location='cpu'))
        model.to(self.device)
        model.eval()
        self.model = model

    def _get_image_list(self, test_data_root):
        """获取图像列表 - 可根据需要重写此方法"""
        img_list = []
        for ext in self.config.IMG_EXTENSIONS:
            img_list.extend(Path(test_data_root).rglob(f'*.{ext}'))
            img_list.extend(Path(test_data_root).rglob(f'*.{ext.upper()}'))
        # 去重并排序
        return sorted({str(P) for P in img_list})

    def _init_ui(self):
        """初始化UI组件"""
        # 按钮定义 (x, y, width, height)
        btn_width = 100
        btn_height = 40
        btn_margin = 10
        start_x = 10
        start_y = 10

        self.btn_prev = (start_x, start_y, btn_width, btn_height)
        self.btn_next = (start_x + btn_width + btn_margin, start_y, btn_width, btn_height)
        self.btn_infer = (start_x + 2 * (btn_width + btn_margin), start_y, btn_width, btn_height)
        self.btn_all = (start_x + 3 * (btn_width + btn_margin), start_y, btn_width, btn_height)

        self.win = 'Inference'
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT)
        cv2.setMouseCallback(self.win, self._mouse)

    # -------------------- 静态工具 --------------------
    @staticmethod
    def _parse_true_label(path, class_names):
        """解析真实标签 - 可根据需要重写此方法"""
        name = os.path.splitext(os.path.basename(path))[0].lower()
        for cls in class_names:
            if name.startswith(cls):
                return cls
        return 'unknown'

    @staticmethod
    def _cv2_imread_rgb(path):
        """读取图像为RGB格式 - 可根据需要重写此方法"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _draw_btn(canvas, txt, rect, color=(200, 200, 200)):
        """绘制按钮"""
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 2)

        # 计算文字位置使其居中
        text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(canvas, txt, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    @staticmethod
    def _resize_image(img, max_width=900, max_height=600):
        """调整图片大小，保持宽高比"""
        h, w = img.shape[:2]

        # 计算缩放比例
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 调整大小
        resized = cv2.resize(img, (new_w, new_h))
        return resized, new_w, new_h

    # -------------------- 推理 --------------------
    def _infer_one(self, path):
        """单张图像推理 - 可根据需要重写此方法"""
        img_rgb = self._cv2_imread_rgb(path)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(logits.argmax(1))
        return pred, prob[pred], img_rgb

    def _infer_current(self):
        """推理当前图像"""
        path = self.img_list[self.idx]
        if path not in self.cache:
            pred, conf, _ = self._infer_one(path)
            self.cache[path] = (pred, conf)

    def _global_infer(self):
        """全局推理"""
        for idx, path in simple_bar(self.img_list, desc='Global inference'):
            if path not in self.cache:
                pred, conf, _ = self._infer_one(path)
                self.cache[path] = (pred, conf)
        self.global_done = True

    # -------------------- 鼠标回调 --------------------
    def _mouse(self, event, x, y, _flags, _param):
        """鼠标事件处理"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        def inside(px, py, rect):
            rx, ry, rw, rh = rect
            return rx <= px <= rx + rw and ry <= py <= ry + rh

        # 只在UI区域内检测点击
        if y < self.config.UI_HEIGHT:
            if inside(x, y, self.btn_prev):
                self.idx = (self.idx - 1) % len(self.img_list)
            elif inside(x, y, self.btn_next):
                self.idx = (self.idx + 1) % len(self.img_list)
            elif inside(x, y, self.btn_infer):
                self._infer_current()
            elif inside(x, y, self.btn_all):
                self._global_infer()

    # -------------------- 处理键盘输入 --------------------
    def _handle_keyboard(self):
        """处理键盘输入"""
        current_time = time.time()

        # 检查是否可以处理按键重复
        can_process = (current_time - self.last_key_time) > self.config.KEY_REPEAT_DELAY

        # 获取按键状态
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):  # Q键退出
            return True
        elif key == ord('a') or key == ord('A') or key == 81:  # 左箭头或A键
            if can_process:
                self.idx = (self.idx - 1) % len(self.img_list)
                self.last_key_time = current_time
        elif key == ord('d') or key == ord('D') or key == 83:  # 右箭头或D键
            if can_process:
                self.idx = (self.idx + 1) % len(self.img_list)
                self.last_key_time = current_time
        elif key == 32:  # 空格键
            if can_process:
                self._infer_current()
                self.last_key_time = current_time

        return False

    # -------------------- 主循环 --------------------
    def run(self):
        """主循环"""
        while True:
            path = self.img_list[self.idx]
            true_label = self._parse_true_label(path, self.class_names)

            if self.global_done or path in self.cache:
                pred_idx, conf = self.cache[path]
                pred_label = self.class_names[pred_idx]
                status_text = f'Pred: {pred_label} ({conf:.2%})  |  True: {true_label}'
                status_color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)
            else:
                status_text = 'Not inferred yet'
                status_color = (200, 200, 200)

            # 读取并调整图片大小
            img_rgb = self._cv2_imread_rgb(path)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_resized, img_w, img_h = self._resize_image(img_bgr,
                                                           max_width=self.config.WINDOW_WIDTH,
                                                           max_height=self.config.WINDOW_HEIGHT - self.config.UI_HEIGHT)

            # 创建画布 - 分隔式UI
            canvas_height = img_h + self.config.UI_HEIGHT
            canvas = np.zeros((canvas_height, self.config.WINDOW_WIDTH, 3), dtype=np.uint8)

            # 放置图片
            canvas[0:img_h, 0:img_w] = img_resized

            # 绘制分隔线
            cv2.line(canvas, (0, img_h), (self.config.WINDOW_WIDTH, img_h), (100, 100, 100), 2)

            # 绘制UI区域背景
            ui_region = canvas[img_h:img_h + self.config.UI_HEIGHT, 0:self.config.WINDOW_WIDTH]
            ui_region.fill(50)  # 深灰色背景

            # 绘制按钮
            self._draw_btn(canvas, 'Prev', self.btn_prev)
            self._draw_btn(canvas, 'Next', self.btn_next)
            self._draw_btn(canvas, 'Infer', self.btn_infer)
            self._draw_btn(canvas, 'Global', self.btn_all)

            # 显示状态信息
            cv2.putText(canvas, status_text, (10, img_h + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # 显示文件名和索引
            file_info = f'{os.path.basename(path)} [{self.idx + 1}/{len(self.img_list)}]'
            cv2.putText(canvas, file_info, (500, img_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 显示操作提示
            hint_text = 'A/D or ←/→: Navigate  Space: Infer  Q: Quit'
            cv2.putText(canvas, hint_text, (500, img_h + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(self.win, canvas)

            # 调整窗口大小以适应图片
            if canvas.shape[1] != self.config.WINDOW_WIDTH or canvas.shape[0] != self.config.WINDOW_HEIGHT:
                self.config.WINDOW_WIDTH = canvas.shape[1]
                self.config.WINDOW_HEIGHT = canvas.shape[0]
                cv2.resizeWindow(self.win, self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT)

            # 处理键盘输入
            if self._handle_keyboard():
                break

        cv2.destroyAllWindows()


# -------------------- 入口 --------------------
if __name__ == '__main__':
    # 使用默认配置
    InferenceUI(test_data_root=Test_data_root, train_data_root=Train_data_root).run()
