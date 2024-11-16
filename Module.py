# PYTHON基础库
import math  # 数学计算库

# 数值计算库
import numpy as np
from numpy import random  # 随机数库

Pi = np.pi

# 绘图库
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文显示
plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号

# 表格处理库
import pandas as pd

# 科学计算库
import scipy
from scipy import signal  # 信号处理库
from scipy import fft  # 傅里叶变换库
