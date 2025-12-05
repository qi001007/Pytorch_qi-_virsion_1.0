"""
simple_bar  ——  不依赖 tqdm 的单行字符进度条
========================================================
功能：
    把任何 **可迭代且支持 len()** 的对象（列表、range、DataLoader …）包一层，
    边迭代边在终端刷出  ████----  样式进度条；
    也可直接传 **整数**，会被当成 range(n) 使用。
    新增「退出标志」支持：把全局标志位传给 flag 参数，一旦标志为 True，进度条会立即
    优雅停止并返回，方便实现“按 q 退出训练”之类的需求。

依赖：
    仅标准库 + numpy（可选）

如何在其他文件调用：
--------------------------------------------------------
1. 把本文件命名成  my_bar.py  扔进你的项目目录
2. 在需要的地方：
--------------------------------------------------------
from my_bar import simple_bar
import time

# ① 传整数
for i in simple_bar(1000, desc="Training", unit="img"):
    ...  # 你的训练/推理逻辑

# ② 传 DataLoader（最常用）—— 支持随时退出
kill_switch = False          # 全局标志，由键盘监听线程置位
for x, y in simple_bar(train_loader, flag=kill_switch,
                       desc="Epoch 1", unit="batch", step_size=50):
    ...  # 训练代码；一旦 kill_switch=True 会优雅停止

# ③ 传普通列表
for row in simple_bar(df.index, desc="Processing", width=40):
    ...  # 处理每行
--------------------------------------------------------
参数说明：
    n          : int / Iterable[T] —— 循环次数或可迭代对象
    flag       : bool —— 外部退出标志（True=立即停），默认 False
    is_open_bar: bool —— 是否显示进度条，默认 True
    desc       : str  —— 进度条前缀文字
    unit       : str  —— 计数单位（img、batch、row…）
    width      : int  —— 进度条宽度（字符数）
    step_size  : int  —— 每隔多少次迭代刷新一次，防闪屏
返回值：
    Generator[T] —— 原样 yield 每个元素，可直接用于 for 循环
"""
import sys
import numpy as np
# 来自 collections.abc.Iterable，是 Python 官方抽象基类
from collections.abc import Iterable, Sized
from typing import cast


def simple_bar(n, flag=False, is_open_bar=True, desc="", unit="", width=30, step_size=50):
    # 1. 如果是“单个数字”，转成 range
    if isinstance(n, (int, np.integer)):
        n = range(n)
    # 2. 判断 n 是否是 Iterable 类型（即能否被 for ... in ... 遍历）,必须是 Iterable + Sized 才能同时 for-in 和 len()
    if not (isinstance(n, Iterable) and isinstance(n, Sized)):
        raise TypeError(f"simple_bar: 需要可迭代对象或整数，当前类型 {type(n).__name__}")
    total = len(n)
    n = cast(Iterable, n)       # 用 typing.cast 告诉检查器：“我保证它同时是可迭代的”

    for idx, item in enumerate(n, 1):
        # 功能1：检测标志位跳出循环
        if flag:  # 全局标志
            print("\n[q] 被按下，将结束训练。")
            raise KeyboardInterrupt  # ✅ 关键：抛出异常，强制中断外层循环
            # break  # 直接停掉生成器
        yield idx, item  # 把元素原样交给 for 循环
        # 功能2： 生成bar
        if is_open_bar:
            # 每 step_size 次刷新一次，最后一定刷新
            if idx % step_size == 0 or idx == total:
                percent = int(100 * idx / total)
                filled = int(width * idx // total)
                bar = '█' * filled + '-' * (width - filled)
                # 关键三行：回车 → 擦行 → 写新内容
                sys.stdout.write('\r')               # 回到行首
                sys.stdout.write('\033[K')           # 擦掉行尾
                sys.stdout.write(f'{desc}: {unit} |{bar}| {percent}%  ({idx}/{total})'.lstrip())
                sys.stdout.flush()
    print()   # 结束时换行
