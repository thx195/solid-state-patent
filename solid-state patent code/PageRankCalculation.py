import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from community import community_louvain

def save_pagerank_csv(pagerank):
    """将原始PageRank值保存为两列CSV文件"""
    # 按PageRank值排序并取前15个国家
    sorted_items = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # 创建DataFrame
    df_output = pd.DataFrame({
        "Country": [k for k, v in sorted_items],
        "PageRank": [v for k, v in sorted_items]
    })
    
    # 保存CSV文件（不包含列名）
    df_output.to_csv("raw_pagerank.csv", index=False, header=False)
    print("CSV文件已生成：raw_pagerank.csv")
def prepare_data():
    df = pd.read_csv('./Dataset/SubResults/CountryCite_Relations.csv')
    total_edges = df.groupby(['CountryCite', 'CountryCitedby'])['Acount'].sum().reset_index()
    node_size = df.groupby('CountryCitedby')['Acount'].sum().reset_index()
    node_size['log_size'] = np.log(node_size['Acount'] + 1)  # 对数标准化
    return df, total_edges, node_size
# ======================
# 主程序修改部分
# ======================
if __name__ == '__main__':
    df, total_edges, node_size = prepare_data()
    
    # 构建图
    G = nx.DiGraph()
    for _, row in total_edges.iterrows():
        G.add_edge(row['CountryCite'], row['CountryCitedby'], weight=row['Acount'])

    # 计算原始PageRank（注意alpha参数保持默认0.85）
    pagerank = nx.pagerank(G, weight='weight')
    
    # 保存结果
    save_pagerank_csv(pagerank)