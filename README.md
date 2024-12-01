# 非线性扩散滤波器算法(Nonlinear Diffusion Filter)

## 开发思路
1. 结合小波分解, 提高待扩散信号的特征强度, 以便更好地进行滤波处理
2. 测试不同扩散特征下的滤波效果, 将多种信号统计特征作为扩散特征选项, 实现不同滤波效果的选择
3. 测试直接卷积迭代格式, 和精确的差分线性方法格式的滤波效果差异
