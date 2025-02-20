import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from community import community_louvain

# ======================
# 数据准备与网络构建
# ======================
def prepare_data():
    df = pd.read_csv('./Dataset/SubResults/CountryCite_Relations.csv')
    total_edges = df.groupby(['CountryCite', 'CountryCitedby'])['Acount'].sum().reset_index()
    node_size = df.groupby('CountryCitedby')['Acount'].sum().reset_index()
    node_size['log_size'] = np.log(node_size['Acount'] + 1)  # 对数标准化
    return df, total_edges, node_size

# ======================
# 可视化函数集
# ======================
def draw_rose(pagerank):
    """改进的玫瑰图可视化，使用对数坐标轴"""
    
    # 取前15个国家
    items = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [k for k, v in items]
    values = np.array([v for k, v in items])
    
    # 数据标准化（归一化到0~100之间，并避免0值）
    max_value = np.max(values)
    min_value = np.min(values)
    values_normalized = (values - min_value) / (max_value - min_value) * 100  # 归一化到0-100
    values_normalized += 1e-9  # 添加极小值避免对数计算问题
    
    # 极坐标参数
    num_labels = len(labels)
    theta = np.linspace(0, 2 * np.pi, num_labels, endpoint=False)
    width = 2 * np.pi / num_labels * 0.9  # 保留间隔
    
    # 开始绘图
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, values_normalized, width=width, alpha=0.8, linewidth=1)
    
    # 颜色映射
    cmap = plt.colormaps.get_cmap('viridis')
    max_norm = np.max(values_normalized)
    for bar, val in zip(bars, values_normalized):
        bar.set_facecolor(cmap(val / max_norm))
    
    # 设置对数坐标轴
    ax.set_yscale('log')
    
    # 坐标轴优化
    ax.set_theta_offset(np.pi / 2)  # 0度方向朝上
    ax.set_theta_direction(-1)      # 顺时针方向
    ax.set_rlabel_position(0)       # 径向标签位置
    
    # 设置刻度标签
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])  # 隐藏径向刻度标签
    
    # 标题和样式
    plt.title("Log-scale Normalized PageRank (Top 15 Countries)", 
             pad=30, fontsize=12, fontweight='bold')
    plt.grid(axis='x', alpha=0.5)
    plt.tight_layout()
    plt.savefig('rose_diagram.png', dpi=300, transparent=True)
    plt.close()

# ======================
# 主程序
# ======================
if __name__ == '__main__':
    # 读取数据
    df, total_edges, node_size = prepare_data()
    
    # 构建图
    G = nx.DiGraph()
    for _, row in total_edges.iterrows():
        G.add_edge(row['CountryCite'], row['CountryCitedby'], weight=row['Acount'])

    # 计算 PageRank
    pagerank = nx.pagerank(G, weight='weight')

    # 绘制玫瑰图
    draw_rose(pagerank)
