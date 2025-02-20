import os
import networkx as nx

def load_and_accumulate_graphs(start_year, end_year, directory):
    # 用来存储每个年份的累积网络
    accumulated_graphs = {}
    
    # 读取和处理每个文件
    for year in range(start_year, end_year + 1):
        accumulated_graph = None
        
        # 遍历当前年份及之前所有年份的网络
        for previous_year in range(1934, year + 1):
            file_name = f"ipc_network_{previous_year}.0.gexf"
            file_path = os.path.join(directory, file_name)
            
            if os.path.exists(file_path):
                # 加载网络
                graph = nx.read_gexf(file_path)
                
                # 如果累积网络不存在，则当前网络为累积网络的开始
                if accumulated_graph is None:
                    accumulated_graph = graph
                else:
                    # 将当前网络合并到累积网络中
                    accumulated_graph = nx.compose(accumulated_graph, graph)
        
        # 如果年份大于等于2008，保存累积网络
        if year >= 2008 and accumulated_graph is not None:
            output_file_name = f"ipc_network_accumulated_{year}.gexf"
            output_file_path = os.path.join(directory, output_file_name)
            nx.write_gexf(accumulated_graph, output_file_path)
            print(f"Accumulated network for {year} saved as {output_file_name}")

# 设置起始和结束年份
START_YEAR = 2008
END_YEAR = 2024

# 设置文件存储的目录
DIRECTORY = "./Results/02ComplexNetworkAnalysis/SeriesNetwork"

# 执行函数
load_and_accumulate_graphs(START_YEAR, END_YEAR, DIRECTORY)
