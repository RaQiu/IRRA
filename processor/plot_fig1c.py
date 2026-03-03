import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 设置论文级绘图样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0

def plot_fig1c(data_path, save_path, epoch_range=(5,30)):
    """
    绘制Fig1(c)：跨模态梯度噪声敏感度对比图
    :param data_path: processor保存的pkl文件路径
    :param save_path: 图片保存路径（支持pdf/png）
    :param epoch_range: 横轴Epoch范围，默认(5,30)匹配论文
    """
    # 1. 读取梯度数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 2. 筛选Epoch范围（5-30）
    epochs = np.array(data["epoch_list"])
    mask = (epochs >= epoch_range[0]) & (epochs <= epoch_range[1])
    plot_epochs = epochs[mask]
    
    # 提取4条曲线数据，缩放×10^-4 匹配纵轴单位
    text_no_noise = np.array(data["text_no_noise"])[mask] * 1e-4
    text_with_noise = np.array(data["text_with_noise"])[mask] * 1e-4
    img_no_noise = np.array(data["img_no_noise"])[mask] * 1e-4
    img_with_noise = np.array(data["img_with_noise"])[mask] * 1e-4

    # 3. 创建1行2列子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
    
    # ========== 左子图：文本模态（浅绿/深绿） ==========
    ax1.plot(plot_epochs, text_no_noise, color='#90EE90', linewidth=2.5, label='Text without noise')
    ax1.plot(plot_epochs, text_with_noise, color='#228B22', linewidth=2.5, label='Text with noise')
    ax1.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'Gradient Value ($\times10^{-4}$)', fontsize=14, fontweight='bold')
    ax1.set_title('Text Modality', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(epoch_range)

    # ========== 右子图：图像模态（浅蓝/深蓝） ==========
    ax2.plot(plot_epochs, img_no_noise, color='#ADD8E6', linewidth=2.5, label='Image without noise')
    ax2.plot(plot_epochs, img_with_noise, color='#1E90FF', linewidth=2.5, label='Image with noise')
    ax2.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax2.set_ylabel(r'Gradient Value ($\times10^{-4}$)', fontsize=14, fontweight='bold')
    ax2.set_title('Image Modality', fontsize=15, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(epoch_range)

    # 4. 保存高清图（论文常用pdf格式，可改为png）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Fig1(c) saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to fig1c_gradient_data.pkl')
    parser.add_argument('--save_path', type=str, default='./fig1c.pdf', help='save path of Fig1(c)')
    parser.add_argument('--epoch_start', type=int, default=5)
    parser.add_argument('--epoch_end', type=int, default=30)
    args = parser.parse_args()
    
    # 执行绘图
    plot_fig1c(
        data_path=args.data_path,
        save_path=args.save_path,
        epoch_range=(args.epoch_start, args.epoch_end)
    )