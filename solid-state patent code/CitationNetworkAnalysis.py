import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from pyvis.network import Network
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
# 复杂网络分析函数集
# ======================
def analyze_network(G):
    metrics = {}
    G_undir = G.to_undirected()
    
    # 基本指标
    metrics['node_count'] = G.number_of_nodes()
    metrics['edge_count'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    metrics['degree_centrality'] = nx.degree_centrality(G)
    metrics['pagerank'] = nx.pagerank(G, weight='weight')
    
    # 连通性指标
    metrics['num_weak_components'] = nx.number_weakly_connected_components(G)
    if nx.is_weakly_connected(G):
        metrics['diameter'] = nx.diameter(G_undir)
        metrics['avg_path_length'] = nx.average_shortest_path_length(G_undir)
    else:
        metrics['diameter'] = np.nan
        metrics['avg_path_length'] = np.nan
    
    # 度相关指标
    degrees = dict(G.degree())
    metrics['avg_degree'] = sum(degrees.values())/metrics['node_count']
    weighted_degrees = dict(G.degree(weight='weight'))
    metrics['avg_weighted_degree'] = sum(weighted_degrees.values())/metrics['node_count']
    
    # 社区检测指标（Louvain方法）
    partition = community_louvain.best_partition(G_undir)
    metrics['modularity'] = community_louvain.modularity(partition, G_undir)
    partition_res = community_louvain.best_partition(G_undir, resolution=1.5)
    metrics['modularity_res'] = community_louvain.modularity(partition_res, G_undir)
    metrics['num_communities'] = len(set(partition.values()))
    
    # 聚类系数
    metrics['clustering'] = nx.average_clustering(G_undir)
    
    # 三角形数量
    triangles = nx.triangles(G_undir)
    metrics['total_triangles'] = sum(triangles.values()) // 3
    
    # 特征向量中心性
    try:
        eigen = nx.eigenvector_centrality(G, max_iter=1000)
        metrics['sum_eigen'] = sum(eigen.values())
    except:
        metrics['sum_eigen'] = np.nan
    
    return metrics

# ======================
# 可视化函数集
# ======================
def draw_network(G, node_size_dict, metrics):
    """使用pyvis生成交互式网络图"""
    partition = community_louvain.best_partition(G.to_undirected())
    colors = px.colors.qualitative.Plotly

    nt = Network(
        height="800px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
    for node in G.nodes():
        community = partition[node]
        nt.add_node(
            n_id=node,
            label=node,
            size=node_size_dict.get(node, 1)*1.5 + 5,  # 缩小节点尺寸
            color=colors[community % len(colors)],
            title=f"""
            Country: {node}<br>
            PageRank: {metrics['pagerank'].get(node, 0):.4f}<br>
            Community: {community}
            """
        )
    
    for u, v, data in G.edges(data=True):
        nt.add_edge(u, v, value=data['weight']**0.5)  # 边宽缩放
    
    nt.set_options(""" 
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -1500,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.05,
          "damping": 0.12,
          "avoidOverlap": 0.2
        },
        "minVelocity": 0.75
      },
      "nodes": {
        "font": {
          "size": 18,
          "strokeWidth": 1
        }
      }
    }
    """)
    
    output_path = 'interactive_network.html'
    nt.write_html(output_path)  # 先写文件
    print(f"Network visualization saved as {output_path}. Please open it in your browser.")

def draw_rose(pagerank):
    """改进的玫瑰图可视化，使用对数坐标轴"""
    # 取前15个国家
    items = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [k for k, v in items]
    values = [v for v in [item[1] for item in items]]
    
    # 数据标准化（比如归一化到0~100之间，避免过大差距）
    max_value = max(values)
    values = [(val / max_value) * 100 for val in values]  # 去掉 eps，直接归一化

    # 极坐标参数
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    width = 2 * np.pi / len(labels) * 0.9  # 保留间隔
    
    # 开始绘图
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, values, width=width, alpha=0.8, linewidth=1)

    # 颜色映射
    cmap = plt.colormaps.get_cmap('viridis')
    for bar, val in zip(bars, values):
        bar.set_facecolor(cmap(val / 100))  # 根据相对值映射颜色
    
    # 设置对数坐标
    ax.set_yscale('log')

    # 坐标轴优化
    ax.set_theta_offset(np.pi / 2)  # 0度方向朝上
    ax.set_theta_direction(-1)  # 顺时针方向
    ax.set_rlabel_position(0)  # 径向标签位置

    # 设置刻度标签
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=10)

    # 隐藏径向轴标签
    ax.set_yticklabels([])

    # 标题和样式
    plt.title("Log-scale Normalized PageRank (Top 15 Countries)", 
             pad=30, fontsize=12, fontweight='bold')
    plt.grid(axis='x', alpha=0.5)
    plt.tight_layout()
    plt.savefig('rose_diagram.png', dpi=300, transparent=True)
    plt.close()

# ======================
# 时间演化分析
# ======================
def temporal_analysis(df):
    df = df[df['YYYY'] >= 1980]
    years = sorted(df['YYYY'].unique())
    evolution_data = []
    
    for year in years:
        temp_df = df[df['YYYY'] <= year]
        edge_df = temp_df.groupby(['CountryCite', 'CountryCitedby'])['Acount'].sum().reset_index()
        
        G = nx.DiGraph()
        for _, row in edge_df.iterrows():
            G.add_edge(row['CountryCite'], row['CountryCitedby'], weight=row['Acount'])
        
        evolution_data.append(nx.pagerank(G, weight='weight'))
    
    evolution_df = pd.DataFrame(evolution_data, index=years)
    main_countries = evolution_df.iloc[-1].nlargest(8).index.tolist()
    
    overall_max = evolution_df.max().max()
    eps = 1e-6
    plt.figure(figsize=(12,6))
    for country in main_countries:
        raw_values = evolution_df[country].values
        norm_values = (raw_values / (overall_max + eps)) + eps
        plt.plot(years, norm_values, lw=2, marker='o', markersize=5, label=country)
    
    plt.yscale('log')
    plt.title('PageRank Evolution (Normalized + Log-scale)', 
              fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('Year', fontsize=12, labelpad=10)
    plt.ylabel('Normalized PageRank (log scale)', fontsize=12, labelpad=10)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', frameon=False)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('temporal_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

# ======================
# 主程序
# ======================
if __name__ == '__main__':
    df, total_edges, node_size = prepare_data()
    node_size_dict = dict(zip(node_size['CountryCitedby'], node_size['log_size']))
    
    G = nx.DiGraph()
    for _, row in total_edges.iterrows():
        G.add_edge(row['CountryCite'], row['CountryCitedby'], weight=row['Acount'])
    
    metrics = analyze_network(G)
    
    indicators = [
        ('Number of Nodes', metrics['node_count']),
        ('Number of Edges', metrics['edge_count']),
        ('Average Degree', metrics['avg_degree']),
        ('Average Weighted Degree', metrics['avg_weighted_degree']),
        ('Diameter', metrics.get('diameter', 'N/A')),
        ('Average Path Length', metrics.get('avg_path_length', 'N/A')),
        ('Density', metrics['density']),
        ('Number of Weakly Connected Components', metrics['num_weak_components']),
        ('Modularity', metrics['modularity']),
        ('Modularity with Resolution', metrics['modularity_res']),
        ('Number of Communities', metrics['num_communities']),
        ('Average Clustering Coefficient', metrics['clustering']),
        ('Total Triangles', metrics['total_triangles']),
        ('Eigenvector Centrality Sum', metrics.get('sum_eigen', 'N/A'))
    ]
    
    pd.DataFrame(indicators, columns=['Metric', 'Value']).to_csv('network_metrics.csv', index=False)
    
    # draw_network(G, node_size_dict, metrics)   # 交互式网络（pyvis）
    draw_rose(metrics['pagerank'])            # 玫瑰图（对数坐标）
    # temporal_analysis(df)                      # PageRank 演化（标准化+对数坐标）
