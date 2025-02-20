import networkx as nx
import pandas as pd
import os

# 创建结果目录
os.makedirs('./Results', exist_ok=True)

# 读取总网络
total_network = nx.read_gexf('./Results/total_ipc_network.gexf')

# 网络基本指标分析
with open('./Results/network_metrics_post_load.txt', 'w') as f:
    f.write(f'Number of Nodes: {total_network.number_of_nodes()}\n')
    f.write(f'Number of Edges: {total_network.number_of_edges()}\n')
    f.write(f'Average Degree: {sum(dict(total_network.degree()).values())/total_network.number_of_nodes()}\n')
    f.write(f'Average Weighted Degree: {sum(dict(total_network.degree(weight="weight")).values())/total_network.number_of_nodes()}\n')
    f.write(f'Diameter: {nx.diameter(total_network) if nx.is_connected(total_network) else "Not Connected"}\n')
    f.write(f'Average Path Length: {nx.average_shortest_path_length(total_network) if nx.is_connected(total_network) else "Not Connected"}\n')
    f.write(f'Density: {nx.density(total_network)}\n')
    f.write(f'Number of Connected Components: {nx.number_connected_components(total_network)}\n')
    f.write(f'Modularity: {"To be calculated if communities are detected"}\n')
    f.write(f'Average Clustering Coefficient: {nx.average_clustering(total_network)}\n')
    f.write(f'Total Triangles: {sum(nx.triangles(total_network).values())//3}\n')
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
node_metrics.to_csv('./Results/node_importance_post_load.csv', index=False)

# 富人俱乐部和度度相关性分析
rich_club_coefficient = nx.rich_club_coefficient(total_network, normalized=False)
degree_assortativity_coefficient = nx.degree_assortativity_coefficient(total_network)
with open('./Results/rich_club_degree_correlation_post_load.txt', 'w') as f:
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
attack_analysis.to_csv('./Results/network_attack_analysis_post_load.csv', index=False)

print("All post-load analyses completed.")
