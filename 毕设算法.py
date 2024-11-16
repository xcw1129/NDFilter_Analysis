import numpy as np
import matplotlib.pyplot as plt


def f_d2(x):
    kernel=[-1/12,4/3,-5/2,4/3,-1/12]#二阶导数的5点差分系数
    return np.convolve(x,kernel,'same')

def Gaussian(x,sigma):#标准高斯函数
    return np.exp(-x**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def diffusion(E,lam,nperseg=None):#扩散控制函数
    if type(lam)!=list:#全局阈值  
        return (1-(E/lam)**2)/(1+0.5*(E/lam)**4)
    else:#局部阈值
        res=[]
        E_seg=np.array_split(E,len(E)/nperseg)#对输入特征进行分割
        for i,seg in enumerate(E_seg):
            res.append((1-(seg/lam[i])**2)/(1+0.5*(seg/lam[i])**4))#分段代入扩散控制函数，每段使用局部阈值
        res=np.concatenate(res)#合并得全局扩散系数
        return res

def PHI(data,lam,G,nperseg=None):
    data=np.square(data)#平方计算方差
    E=np.convolve(data,G,'same')#卷积得信号功率流，也即时变方差
    res=diffusion(E,lam,nperseg)#局部阈值下的扩散系数或全局阈值下的扩散系数，取决于lam的类型
    return res

def plot_nonlineardiffusion(_data,_fs,window,kappa=2,iterations=100,dt=0.2,title=None,result=False):
    """
    二阶自适应扩散滤波

    参数
    ----
    _data:np.array
        待扩散滤波的时序数据
    _fs:float
        采样频率
    window:float
        高斯卷积核窗长
    kappa:float
        局部阈值倍数，默认为2
    iterations:int
        迭代次数，默认为100
    dt:float
        迭代步长，默认为0.2
    title:str
        图像标题，默认为None
    result:bool
        是否返回滤波后的数据，默认为False

    返回
    ----
    result:np.array
        滤波后的数据
    """
    data=_data.copy()

    #根据window生成高斯卷积核，在扩散中保持不变
    time=np.arange(0,window/2,1/_fs)#高斯平滑窗单边长度为3倍标准差
    time=np.append(-time[::-1],time[1:])#生成时间序列
    G = Gaussian(time,window/6)  # 生成高斯核函数
    G = G / np.sum(G)  # 归一化

    #根据原始信号计算局部阈值，在扩散中保持不变
    E=np.convolve(np.square(data),G,'same')
    threshold=[]
    nperseg=int(3*window*_fs)#分段阈值，每段长度3倍卷积核窗长
    E_seg=np.array_split(E,len(E)//nperseg)
    for seg in E_seg:
        threshold.append(kappa*seg.mean())#计算局部阈值，取为段内方差平均值的2倍使得缓变趋势不被增强，瞬变趋势得到增强

    #迭代扩散
    for i in range(iterations):
        p=PHI(data,threshold,G,nperseg)#扩散系数
        d2=f_d2(data)#二阶导数
        if i!=0:
            inc=np.std(d2*p)-np.std(delta)#增量收敛判断
            if np.abs(inc/delta.std())<0.01:#扩散增量收敛判断
                delta=d2*p
                data=delta*dt+data#最后一次迭代
                break#增量收敛则跳出迭代
        delta=d2*p#扩散增量
        data=delta*dt+data#迭代计算,dt=0.2为标准步长

    print(f"迭代{i}次")

    #绘制扩散后时域波形    
    plt.figure(figsize=(12, 5))
    plt.plot(np.linspace(0,len(data)/_fs,len(data)), data)
    plt.title(title)
    plt.show()
    if result:
        return data
    else:
        return None
    
def find_impacts(_data,_fs,width):
    """
    定步长查找冲击，根据信号局部标准差判定冲击

    参数
    ----
    _data:np.array
        待查找冲击的时序数据
    _fs:float
        采样频率
    width:float
        冲击波宽度

    返回
    ----
    impact_time:np.array
        冲击时刻序列,每行为一个冲击的起始和结束时刻
    """

    door=_data.std()*3
    step=width/5#步长为冲击波宽度的1/5
    index = np.arange(0, len(_data), int(step*_fs))
    index = index.astype(int)#查找起点索引
    windowsize=2*step#窗长为2倍步长
    nperseg = int(windowsize*_fs)

    impact_time = []  # 冲击时刻序列(a,b)，a为冲击波起始时刻，b为冲击结束时刻
    impactflag = 0  # 冲击标志
    start, stop = 0, 0  # 冲击起始和结束时刻

    count=0#冲击中断计数

    for i in index:
        if i + nperseg > len(_data):  # 防止越界
            data = _data[i:]#不足窗长，取到末尾
            break
        else:
            data = _data[i:i + nperseg]  # 按步长取信号片段，窗长0.01s等于2倍冲击波宽度

        if data.std() > door:  # 检测到异常
            if impactflag == 0:  # 冲击开始
                start = i/_fs#以窗口起点作为冲击起始时刻
                impactflag = 1
            count=0#冲击中断计数清零
        else:
            if impactflag == 1:  # 冲击中断
                if count>=10:#冲击中断时间大于阈值，认为冲击结束
                    stop = i/_fs#以窗口起点作为冲击结束时刻
                    impact_time.append((start, stop))  # 记录冲击
                    impactflag = 0#重置冲击标志，准备查找下一个冲击
                    count=0
                else:
                    count+=1#冲击中断计数
            else:
                continue#无冲击，继续查找
                
        
    if impactflag==1:#检查信号尾部存在未结束的冲击
        stop=len(_data)/_fs#以信号末尾作为冲击结束时刻
        impact_time.append((start,stop))

    impact_time = np.array(impact_time)
    return impact_time

#递归查找冲击链，默认不存在起始冲击相同的冲击链
def detect_impactclain(interval,delta):
    """
    检测冲击链，冲击链内冲击间隔在理想冲击间隔倍数加减容差范围内

    参数
    ----
    interval:np.array
        冲击间隔序列
    delta:list
        冲击链的冲击间隔,delta[0]为理想冲击间隔，delta[1]为冲击间隔容差

    返回
    ----
    clain:np.array
        冲击链序列
    """
    T=delta[0]#理想冲击间隔
    sigma=delta[1]#冲击间隔容差

    clain=np.array([1],dtype=int)#冲击链记录
    dot=np.array([-999],dtype=int)#冲击间隔记录(用于递归)    
    sum=0#冲击间隔累加

    if len(interval)==1:#递归到最后两个冲击
        if interval[0]%T<=sigma and interval[0]//T>0:#该冲击间隔在容差右侧
            clain=np.append(clain,2)
        elif interval[0]%T>=T-sigma:#该冲击间隔在容差左侧
            clain=np.append(clain,2)
        else:
            clain=np.append(clain,-999)
            clain=np.append(clain,2)
        return clain#冲击链检测结束
    
    for i,t in enumerate(interval):#遍历冲击间隔
        sum+=t#累加冲击间隔
        if sum%T<=sigma and sum//T>0:#该冲击间隔在容差右侧
            sum=0
            clain=np.append(clain,i+2)
        elif sum%T>=T-sigma:#该冲击间隔在容差左侧
            sum=0
            clain=np.append(clain,i+2)
        else:
            continue

    return np.concatenate((clain,dot,detect_impactclain(interval[1:],delta)+1))#将首冲击去除，递归检测下一起点的冲击链