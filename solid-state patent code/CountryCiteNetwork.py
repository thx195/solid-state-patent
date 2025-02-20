import os
import glob
import pandas as pd
import re

def extract_country(patent_number):
    """提取专利号中的国家代码（开头的连续字母）"""
    if pd.isnull(patent_number) or patent_number == '-':
        return None
    patent_str = str(patent_number)
    match = re.match(r'^([A-Za-z]+)', patent_str)
    return match.group(1).upper() if match else None

def process_files():
    # 路径设置
    input_dir = './Dataset/NewData'
    output_dir = './Dataset/SubResults'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # 遍历所有Excel文件
    for file_path in glob.glob(os.path.join(input_dir, '*.xlsx')):
        print(f"正在处理文件: {file_path}")  # 输出正在处理的文件
        
        df = pd.read_excel(
            file_path,
            usecols=['公开号', '引用的参考文献 - 专利', '申请日期'],
            header=1  # 跳过第一行，第二行作为列名
        )
        
        for _, row in df.iterrows():
            # 提取引用国家
            citing_patent = row['公开号']
            citing_country = extract_country(citing_patent)
            if not citing_country:
                continue
            
            # 提取年份
            try:
                year = pd.to_datetime(row['申请日期']).year
            except:
                continue
            
            # 处理被引专利
            refs = row['引用的参考文献 - 专利']
            if pd.isnull(refs) or refs == '-':
                continue
            
            for cited_patent in refs.split(' | '):
                cited_country = extract_country(cited_patent.strip())
                if cited_country:
                    results.append({
                        'CountryCite': citing_country,
                        'CountryCitedby': cited_country,
                        'YYYY': year
                    })
    
    # 聚合结果并保存
    if results:
        df_result = pd.DataFrame(results)
        df_grouped = df_result.groupby(
            ['CountryCite', 'CountryCitedby', 'YYYY'], 
            as_index=False
        ).size().rename(columns={'size': 'Acount'})
        
        output_path = os.path.join(output_dir, 'CountryCite_Relations.csv')
        df_grouped.to_csv(output_path, index=False)
        print(f'结果已保存至: {output_path}')
    else:
        print("未找到有效数据")

if __name__ == '__main__':
    process_files()
