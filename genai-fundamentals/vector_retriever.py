import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings  # 替换为BGE-M3

load_dotenv()

# 连接 Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

# 使用 bge-m3 embedding 模型（本地/HuggingFace Hub）
embedder = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",  # intfloat/e5-large-v2 模型 --- 1536 维
    model_kwargs={"device": "cpu"},  # 如果有GPU可改成 "cuda"
    encode_kwargs={"normalize_embeddings": True}  # 保证向量归一化
)

# 创建向量检索器
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# 搜索
result = retriever.search(query_text="Toys coming alive", top_k=5)

# 输出结果
for item in result.items:
    print(item.content, item.metadata["score"])

# 关闭连接
driver.close()
