from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os

# MongoDB连接设置
client = MongoClient('localhost', 27017)
db = client['PatentData']
collection = db['AIPatentData']

# 初始化LDA模型和词频向量器
n_topics = 10
batch_size = 2000  # 批处理大小
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online')

# 创建结果保存目录
os.makedirs('./Results', exist_ok=True)

# 第一阶段：确定固定词汇表
cursor = collection.find()
batch_docs = []

for doc in cursor:
    if isinstance(doc['摘要 (英语)'], str):
        batch_docs.append(doc['摘要 (英语)'])

    if len(batch_docs) >= batch_size:
        break  # 只用第一批文档生成词汇表

# 使用第一批文档生成词汇表
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
vectorizer.fit(batch_docs)
vocabulary = vectorizer.vocabulary_

# 第二阶段：处理整个数据集
cursor = collection.find()
batch_docs = []
doc_ids = []

for doc in cursor:
    doc_id = doc['_id']

    # # 统计标题单词数量
    # title_word_count = len(doc['标题 (英语)'].split()) if isinstance(doc['标题 (英语)'], str) else 0
    # collection.update_one({'_id': doc_id}, {'$set': {'标题单词数量': title_word_count}})
    
    # # 统计发明人数量
    # inventor_count = doc['发明人'].count(' | ') + 1 if isinstance(doc['发明人'], str) else 0
    # collection.update_one({'_id': doc_id}, {'$set': {'发明人数量': inventor_count}})
    
    # # 统计IPC 大类数量
    # ipc_count = doc['IPC 大类组'].count(', ') + 1 if isinstance(doc['IPC 大类组'], str) else 0
    # collection.update_one({'_id': doc_id}, {'$set': {'IPC 大类数量': ipc_count}})
    
    # # 统计DWPI 同族专利成员数量
    # dwpi_count = doc['DWPI 同族专利成员'].count(' | ') + 1 if isinstance(doc['DWPI 同族专利成员'], str) else 0
    # collection.update_one({'_id': doc_id}, {'$set': {'DWPI 同族专利成员数量': dwpi_count}})
    
    # # 统计权利要求数量
    # claim_count = doc['权利要求 (英语)'].count(' | ') + 1 if isinstance(doc['权利要求 (英语)'], str) else 0
    # collection.update_one({'_id': doc_id}, {'$set': {'权利要求 (英语)数量': claim_count}})
    
    # 收集摘要 (英语) 文本用于LDA处理
    if isinstance(doc['摘要 (英语)'], str):
        batch_docs.append(doc['摘要 (英语)'])
        doc_ids.append(doc_id)

    # 每当达到批处理大小时，进行LDA处理
    if len(batch_docs) >= batch_size:
        term_matrix = vectorizer.transform(batch_docs)
        lda.partial_fit(term_matrix)
        
        # 更新LDA主题到数据库
        topic_distributions = lda.transform(term_matrix)
        for doc_id, topic_dist in zip(doc_ids, topic_distributions):
            lda_topics = {f'LDA{i+1}': float(topic) for i, topic in enumerate(topic_dist)}
            collection.update_one({'_id': doc_id}, {'$set': lda_topics})
        
        # 清理批处理数据
        batch_docs = []
        doc_ids = []

# 处理剩余的摘要数据
if batch_docs:
    term_matrix = vectorizer.transform(batch_docs)
    lda.partial_fit(term_matrix)
    
    # 更新LDA主题到数据库
    topic_distributions = lda.transform(term_matrix)
    for doc_id, topic_dist in zip(doc_ids, topic_distributions):
        lda_topics = {f'LDA{i+1}': float(topic) for i, topic in enumerate(topic_dist)}
        collection.update_one({'_id': doc_id}, {'$set': lda_topics})

# 保存LDA主题词和每个主题中的单词概率
with open('./Results/LDA_Topics.txt', 'w') as f:
    for i, topic in enumerate(lda.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
        f.write(f'Topic {i+1}: {" ".join(topic_words)}\n')

print("数据处理与分析完成")
