import pandas as pd
import os

# 设置数据文件夹路径
data_folder = './OriginalData'

# 获取所有Excel文件的路径
excel_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.xlsx')]

# 初始化一个空的DataFrame来存储合并后的数据
merged_data = pd.DataFrame()

# 遍历文件路径列表，读取每个Excel文件
for file_path in excel_files:
    # 读取Excel文件，跳过第一行
    df = pd.read_excel(file_path, skiprows=1)
    
    # 如果merged_data为空，则直接赋值
    if merged_data.empty:
        merged_data = df
    else:
        # 按列名合并数据
        merged_data = pd.concat([merged_data, df], ignore_index=True)

# 删除完全重复的行
merged_data = merged_data.drop_duplicates()

# 写入新的Excel文件
output_file = './Dataset/merged_data.xlsx'
merged_data.to_excel(output_file, index=False)

print(f'数据已合并并写入到 {output_file}')
