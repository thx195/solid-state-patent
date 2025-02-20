from pymongo import MongoClient
import pandas as pd
import networkx as nx
import os
from datetime import datetime
from collections import defaultdict

# MongoDB连接设置
client = MongoClient('localhost', 27017)
db = client['PatentData']
collection = db['AIPatentData']

# 创建结果目录
os.makedirs('./Results', exist_ok=True)

# 数据读取和预处理
data = pd.DataFrame(list(collection.find({}, {'IPC 大类组': 1, '申请日期': 1, '申请国家/地区': 1})))
data = data.dropna(subset=['IPC 大类组', '申请日期'])
data['申请日期'] = pd.to_datetime(data['申请日期'], errors='coerce')
data['Year'] = data['申请日期'].dt.year
data['IPC 大类组'] = data['IPC 大类组'].str.split(', ')

# 初始化变量
yearly_networks = defaultdict(nx.Graph)

# 构建时序IPC共现网络
for index, row in data.iterrows():
    ipc_list = row['IPC 大类组']
    year = row['Year']
    
    if len(ipc_list) > 1:
        for i in range(len(ipc_list)):
            for j in range(i + 1, len(ipc_list)):
                if yearly_networks[year].has_edge(ipc_list[i], ipc_list[j]):
                    yearly_networks[year][ipc_list[i]][ipc_list[j]]['weight'] += 1
                else:
                    yearly_networks[year].add_edge(ipc_list[i], ipc_list[j], weight=1)

# 保存时序网络
for year, graph in yearly_networks.items():
    nx.write_gexf(graph, f'./Results/ipc_network_{year}.gexf')

# 生成总网络
total_network = nx.Graph()
for year, graph in yearly_networks.items():
    for u, v, data in graph.edges(data=True):
        if total_network.has_edge(u, v):
            total_network[u][v]['weight'] += data['weight']
        else:
            total_network.add_edge(u, v, weight=data['weight'])

# 保存总网络
nx.write_gexf(total_network, './Results/total_ipc_network.gexf')

# 时序分析和网络基本指标分析
with open('./Results/network_metrics.txt', 'w') as f:
    for year, graph in yearly_networks.items():
        f.write(f'Year: {year}\n')
        f.write(f'Number of Nodes: {graph.number_of_nodes()}\n')
        f.write(f'Number of Edges: {graph.number_of_edges()}\n')
        f.write(f'Average Degree: {sum(dict(graph.degree()).values())/graph.number_of_nodes()}\n')
        f.write(f'Average Weighted Degree: {sum(dict(graph.degree(weight="weight")).values())/graph.number_of_nodes()}\n')
        f.write(f'Diameter: {nx.diameter(graph) if nx.is_connected(graph) else "Not Connected"}\n')
        f.write(f'Average Path Length: {nx.average_shortest_path_length(graph) if nx.is_connected(graph) else "Not Connected"}\n')
        f.write(f'Density: {nx.density(graph)}\n')
        f.write(f'Number of Weakly Connected Components: {nx.number_weakly_connected_components(graph)}\n')
        f.write(f'Modularity: {"To be calculated if communities are detected"}\n')
        f.write(f'Average Clustering Coefficient: {nx.average_clustering(graph)}\n')
        f.write(f'Total Triangles: {sum(nx.triangles(graph).values())//3}\n')
        f.write(f'---\n')

# 节点重要性分析
node_metrics = pd.DataFrame({
    'Node': list(total_network.nodes),
    'Degree': [val for (node, val) in total_network.degree()],
    'Betweenness': [val for (node, val) in nx.betweenness_centrality(total_network).items()],
    'Eigenvector': [val for (node, val) in nx.eigenvector_centrality(total_network).items()],
    'Pagerank': [val for (node, val) in nx.pagerank(total_network).items()],
    'Corenum': [val for (node, val) in nx.core_number(total_network).items()],
    'Closeness': [val for (node, val) in nx.closeness_centrality(total_network).items()]
})
node_metrics.to_csv('./Results/node_importance.csv', index=False)

# 富人俱乐部和度度相关性分析
rich_club_coefficient = nx.rich_club_coefficient(total_network, normalized=False)
degree_assortativity_coefficient = nx.degree_assortativity_coefficient(total_network)
with open('./Results/rich_club_degree_correlation.txt', 'w') as f:
    f.write(f'Rich Club Coefficient: {rich_club_coefficient}\n')
    f.write(f'Degree Assortativity Coefficient: {degree_assortativity_coefficient}\n')

# 网络攻击分析
largest_cc_ratio = []
second_largest_cc_ratio = []

for i in range(1, len(total_network.nodes)):
    temp_network = total_network.copy()
    nodes_to_remove = sorted(temp_network.degree, key=lambda x: x[1], reverse=True)[:i]
    temp_network.remove_nodes_from([n[0] for n in nodes_to_remove])
    largest_cc = max(nx.connected_components(temp_network), key=len)
    largest_cc_ratio.append(len(largest_cc) / total_network.number_of_nodes())
    if len(temp_network) > 1:
        second_largest_cc = max(nx.connected_components(temp_network), key=len)
        second_largest_cc_ratio.append(len(second_largest_cc) / total_network.number_of_nodes())
    else:
        second_largest_cc_ratio.append(0)

attack_analysis = pd.DataFrame({
    'Nodes Removed': range(1, len(total_network.nodes)),
    'Largest Connected Component Ratio': largest_cc_ratio,
    'Second Largest Connected Component Ratio': second_largest_cc_ratio
})
attack_analysis.to_csv('./Results/network_attack_analysis.csv', index=False)

print("All stages completed.")
