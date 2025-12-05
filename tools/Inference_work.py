import os
import cv2
import torch
import argparse
import numpy as np
import time
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import Residual, ResNet18
from my_bar import simple_bar


class InferenceConfig:
    """配置类，便于替换模型、数据集和数据处理方法"""
    # 模型配置
    MODEL_CLASS = ResNet18
    MODEL_ARGS = (Residual,)  # 模型初始化参数
    MODEL_WEIGHTS_PATH = './model_wts/2025-11-29-16-35model/best_model_2025-11-29-16-50.pth'

    # 数据预处理配置
    IMAGE_SIZE = (224, 224)
    MEAN = [0.17263502, 0.15147281, 0.14267276]
    STD = [0.07360361, 0.06215812, 0.05929757]

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
    def __init__(self, img_dir, train_data_root, config=InferenceConfig):
        self.config = config

        # 获取类别名称
        train_set = ImageFolder(root=train_data_root, transform=transforms.ToTensor())
        self.class_names = train_set.classes
        print('=> 训练集类别顺序:', self.class_names)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._load_model()

        # 获取图像列表
        self.img_list = self._get_image_list(img_dir)
        assert self.img_list, f'No image found in {img_dir}'
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
        model = self.config.MODEL_CLASS(*self.config.MODEL_ARGS)
        model.load_state_dict(torch.load(self.config.MODEL_WEIGHTS_PATH, map_location='cpu'))
        model.to(self.device)
        model.eval()
        self.model = model

    def _get_image_list(self, img_dir):
        """获取图像列表 - 可根据需要重写此方法"""
        img_list = []
        for ext in self.config.IMG_EXTENSIONS:
            img_list.extend(Path(img_dir).rglob(f'*.{ext}'))
            img_list.extend(Path(img_dir).rglob(f'*.{ext.upper()}'))
        # 去重并排序
        return sorted({str(p) for p in img_list})

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='./data/test', help='folder contains test images')
    parser.add_argument('--train_dir', default='./data/train', help='folder used for training')
    args = parser.parse_args()

    # 使用默认配置
    InferenceUI(img_dir=args.img_dir, train_data_root=args.train_dir).run()
