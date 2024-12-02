import numpy as np

from PySP.Signal import Signal, Analysis
from PySP.Plot import plot_spectrum, plot_2D_Anim

import matplotlib.pyplot as plt
from matplotlib import font_manager

zh_font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")  # 中文字体

FLOAT_EPS = 1e-6


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class NDfilter_Analysis(Analysis):
    @Analysis.Input({"Sig": {}})
    def __init__(
        self,
        Sig: Signal,
        CvgError: float = 0.2,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # ------------------------------------------------------------------------------------#
        self.CvgError = CvgError  # 收敛误差: 误差使用L2范数

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_spectrum)
    @Analysis.Input(
        {"dt": {"CloseHigh": 0.5, "CloseLow": 0.01}, "iters": {"High": 200, "Low": 10}}
    )
    def LDF(self, dt: float, iters: int):
        # 初始化
        data = self.Sig.data.copy()
        N = self.Sig.N
        # 边界条件处理
        left = np.mean(data[:100])
        right = np.mean(data[-100:])
        # ------------------------------------------------------------------------------------#
        # 隐式差分-τU_(i-1,j)+(1+2τ)U_(i,j)-τU_(i+1,j)=U_(i,j-1), 边界条件U_(0,j)=U_(N-1,j)=原始信号边界估计
        for k in range(iters):
            data[0] = left
            data[-1] = right
            _data = data.copy()  # 保持单次迭代前的数据
            # 追赶法求解原差分等价三对角线性方程组: A*U_(i,j+1)=U_(i,j)
            l = np.zeros(N)
            u = np.zeros(N)
            # 求解L*y=U_(i,j)得y
            y = np.zeros(N)
            u[0] = 1
            y[0] = data[0]
            for i in range(1, N):
                l[i] = -dt / u[i - 1]
                u[i] = 1 + 2 * dt - l[i] * (-dt)
                y[i] = data[i] - l[i] * y[i - 1]
            # 求解U*U_(i,j+1)=y得U_(i,j+1)
            data[-1] = y[-1] / u[-1]
            for i in range(N - 2, -1, -1):
                data[i] = (y[i] - (-dt) * data[i + 1]) / u[i]
            # 收敛判断
            ErrorNorm = np.linalg.norm(data - _data)
            if ErrorNorm < self.CvgError:
                break
        return self.Sig.t_Axis, data

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_spectrum)
    @Analysis.Input(
        {
            "WinSize": {"CloseHigh": 0.5, "CloseLow": 0.01},
            "SegNum": {"High": 200, "Low": 10},
            "kappa": {"CloseHigh": 0.5, "CloseLow": 0.01},
            "iters": {"High": 200, "Low": 10},
            "dt": {"CloseHigh": 0.5, "CloseLow": 0.01},
        }
    )
    def SA_NDF(self, WinSize: float, SegNum, kappa: float, iters: int, dt: float):
        # 初始化
        data = self.Sig.data
        fs = self.Sig.fs
        filted_data = data.copy()  # 迭代过程中的滤波信号
        Delta = _Delta = np.zeros_like(data)
        G_win = NDfilter_Analysis.Gaussian(WinSize, int(WinSize * fs))
        # 计算扩散迭代过程的局部阈值
        _, thre = self.PowerFlow(WinSize * 10, SegNum)
        thre *= kappa
        for i in range(iters):
            # 计算扩散控制特征, 该特征决定扩散方向
            DiffFea = np.convolve(np.square(filted_data), G_win, "same")
            # 代入特征到扩散控制函数，得到缩放后的扩散系数
            Coe = NDfilter_Analysis.DiffusionScaler(DiffFea, thre)
            if i % 10 == 0:
                NDfilter_Analysis.plot_2lines(
                    self.Sig.t_Axis, DiffFea, thre, title=f"第{i+1}次迭代特征缩放情况"
                )
                plot_spectrum(self.Sig.t_Axis, Coe, title=f"第{i+1}次迭代扩散系数")
            # 扩散方程增量式迭代
            D2_data = self.Div2(filted_data)  # 信号二阶导
            Delta = Coe * D2_data * dt  # 单步增量
            filted_data += Delta  # 迭代
            # 收敛判断
            ErrorNorm = np.linalg.norm(Delta - _Delta)
            if ErrorNorm < self.CvgError:
                break
            _Delta = Delta.copy()

        print(f"迭代次数:{i+1}")
        return self.Sig.t_Axis, filted_data

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_spectrum)
    @Analysis.Input(
        {
            "WinSize": {"CloseHigh": 0.5, "CloseLow": 0.01},
            "SmoothThre:int": {"High": 200, "Low": 10},
            "kappa": {"CloseHigh": 5, "CloseLow": 1},
            "dt": {"CloseHigh": 0.5, "CloseLow": 0.01},
        }
    )
    def debug_SA_NDF(self, WinSize: float, SmoothThre: int, kappa: float, dt: float):
        # 初始化
        data = self.Sig.data
        fs = self.Sig.fs
        t_Axis = self.Sig.t_Axis
        filted_data = data.copy()  # 迭代过程中的滤波信号
        Delta = _Delta = np.zeros_like(data)
        G_win = NDfilter_Analysis.Gaussian(WinSize, int(WinSize * fs))
        # 计算扩散迭代过程的局部阈值
        _, thre = self.PowerFlow(G_win, SmoothThre)
        thre *= kappa
        # debug输出
        filted_data_process = []
        # ------------------------------------------------------------------------------------#
        # 迭代求解非线性扩散方程
        for i in range(100):
            # 计算扩散控制特征, 该特征决定扩散方向
            DiffFea = np.convolve(np.square(filted_data), G_win, "same")
            # 代入特征到扩散控制函数，得到缩放后的扩散系数
            Coe = NDfilter_Analysis.DiffusionScaler(
                DiffFea, thre
            )  # thre为数组, 即局部阈值
            # 扩散方程增量式迭代
            D2_filted_data = self.Div2(filted_data)  # 信号二阶导
            Delta = Coe * D2_filted_data * dt  # 单步增量
            filted_data += Delta  # 迭代
            # --------------------------------------------------------------------------------#
            # debug输出
            filted_data_process.append(filted_data.copy())
            if i % 5 == 0:
                NDfilter_Analysis.plot_2lines(
                    t_Axis,
                    DiffFea,
                    thre,
                    title=f"第{i+1}次扩散迭代的特征控制情况",
                    legend=("控制特征", "局部阈值"),
                )
            # --------------------------------------------------------------------------------#
            # 收敛判断
            ErrorNorm = np.linalg.norm(Delta - _Delta)
            if ErrorNorm < self.CvgError:
                break
            else:
                # debug输出
                print(f"第{i+1}次迭代增量收敛范数: {np.round(ErrorNorm, 4)}")
            _Delta = Delta.copy()
        # ------------------------------------------------------------------------------------#
        # debug输出
        filted_data_process = np.array(filted_data_process)
        framelabel = [f"第{i+1}次迭代" for i in range(filted_data_process.shape[0])]
        plot_2D_Anim(
            t_Axis,
            filted_data_process,
            xlabel="时间t(s)",
            ylabel="信号幅值",
            title="非线性扩散滤波过程",
            framelabel=framelabel,
        )
        self.plot = True
        print("迭代完成, 扩散过程动画已保存")
        # ------------------------------------------------------------------------------------#
        return self.Sig.t_Axis, filted_data

    # ----------------------------------------------------------------------------------------#
    def PowerFlow(self, window: np.ndarray, n: int):
        # 初始化
        data = self.Sig.data
        # 计算功率流
        G_win = NDfilter_Analysis.Gaussian(size, int(size * fs))
        power = np.convolve(np.square(data), G_win, mode="same")  # 窗卷积计算功率流
        # 功率流分段平均化
        segstart_idx = np.linspace(0, len(power), n + 1, dtype=int)[:-1]  # 分段起始索引
        segend_idx = np.linspace(0, len(power), n + 1, dtype=int)[1:]  # 分段结束索引
        segmean_power = np.zeros_like(power)
        for i in range(n):
            power_seg = power[segstart_idx[i] : segend_idx[i]]
            segmean_power[segstart_idx[i] : segend_idx[i]] = np.mean(
                power_seg, keepdims=True
            )
        return self.Sig.t_Axis, segmean_power

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def DiffusionScaler(x: np.ndarray, lam: np.ndarray):
        return (1 - (x / lam) ** 2) / (1 + 0.5 * (x / lam) ** 4)

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def Div2(data: np.ndarray):
        kernel = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]  # 二阶导数的5点差分系数
        return np.convolve(data, kernel, "same")

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def Gaussian(T: float, N: int):
        if N % 2 == 0:
            t_Axis = np.linspace(-T / 2, T / 2, N, endpoint=False)
        else:
            t_Axis = np.linspace(-T / 2, T / 2, N, endpoint=True)
        sigma = (t_Axis[-1] - t_Axis[0]) / 2
        window = np.exp(-(t_Axis**2) / (2 * sigma**2))
        window /= np.sum(window)  # 归一化
        return window

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def plot_2lines(
        Axis: np.ndarray,
        data1: np.ndarray,
        data2: np.ndarray,
        **kwargs,
    ) -> None:
        """
        根据输入的横轴坐标, 和两列纵轴数据, 绘制双线Plot型谱

        参数:
        ----------
        Axis : np.ndarray
            x轴数据
        data1 : np.ndarray
            y轴数据1
        data2 : np.ndarray
            y轴数据2
        (xlabel) : str, 可选
            x轴标签, 默认为None
        (xticks) : list, 可选
            x轴刻度, 默认为None
        (xlim) : tuple, 可选
            x轴刻度范围, 默认为None
        (ylabel) : str, 可选
            y轴标签, 默认为None
        (ylim) : tuple, 可选
            y轴刻度范围, 默认为None
        (title) : str, 可选
            图像标题, 默认为None
        (plot_save) : bool, 可选
            是否将绘图结果保存为svg图片, 默认不保存
        """
        # 检查数据
        if len(Axis) != len(data1) or len(Axis) != len(data2):
            raise ValueError("输入数据长度不一致")
        # ---------------------------------------------------------------------------------------#
        # 设置图像界面
        figsize = kwargs.get("figsize", (12, 5))
        plt.figure(figsize=figsize)
        # 设置线条格式
        line1 = kwargs.get("line1", "b")
        line2 = kwargs.get("line2", "r--")
        plt.plot(Axis, data1, line1)
        plt.plot(Axis, data2, line2)
        # 设置标题
        title = kwargs.get("title", None)
        plt.title(title, fontproperties=zh_font)
        # 设置图像栅格
        plt.grid(axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10))
        # ---------------------------------------------------------------------------------------#
        # 设置坐标轴参数
        # 设置x轴参数
        xlabel = kwargs.get("xlabel", None)
        plt.xlabel(xlabel, fontproperties=zh_font, labelpad=0.2, loc="right")  # 标签
        xticks = kwargs.get("xticks", None)
        plt.xticks(xticks)  # 刻度显示
        xlim = kwargs.get("xlim", (None, None))
        plt.xlim(xlim[0], xlim[1])  # 刻度范围
        # 设置y轴参数
        ylabel = kwargs.get("ylabel", None)
        plt.ylabel(ylabel, fontproperties=zh_font, labelpad=0.2, loc="top")  # 标签
        ylim = kwargs.get("ylim", (None, None))
        plt.ylim(ylim[0], ylim[1])  # 刻度范围
        # ---------------------------------------------------------------------------------------#
        # 按指定格式保存图片并显示
        plot_save = kwargs.get("plot_save", False)
        if plot_save:
            plt.savefig(title + ".svg", format="svg")  # 保存图片
        plt.show()
