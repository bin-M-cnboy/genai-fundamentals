import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
import uuid

load_dotenv()

# ===============================
# 1️⃣ 连接 Neo4j
# ===============================
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# ===============================
# 2️⃣ 初始化 1536 维 embedding 模型
# ===============================
embedder = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",  # 1536 维
    model_kwargs={"device": "cpu"},      # 或 "cuda"
    encode_kwargs={"normalize_embeddings": True}
)

# ===============================
# 3️⃣ 给节点生成 UUID（只做一次）
# ===============================
with driver.session() as session:
    session.run("""
    MATCH (n:Movie) 
    WHERE n.uuid IS NULL
    SET n.uuid = randomUUID()
    """)

# ===============================
# 4️⃣ 批量更新节点向量到新属性 embedding_1536
# ===============================
BATCH_SIZE = 50
with driver.session() as session:
    offset = 0
    while True:
        rows = list(session.run(
            "MATCH (n:Movie) RETURN n.uuid AS uuid, n.plot AS text SKIP $skip LIMIT $limit",
            skip=offset, limit=BATCH_SIZE
        ))
        if not rows:
            break

        for r in rows:
            node_uuid = r["uuid"]
            text = r["text"] or ""
            vec = embedder.embed_query(text)
            session.run(
                "MATCH (n:Movie) WHERE n.uuid = $uuid SET n.embedding_1536 = $vec",
                uuid=node_uuid,
                vec=vec
            )

        offset += BATCH_SIZE
        print(f"Updated {offset} nodes...")

# ===============================
# 5️⃣ 删除旧索引并创建新向量索引
# ===============================
with driver.session() as session:
    # 删除旧索引 moviePlots（如果存在）
    session.run("""
    CALL db.indexes() YIELD name
    WHERE name = 'moviePlots'
    CALL db.index.drop(name)
    RETURN name
    """)

    # 创建新的向量索引
    session.run("""
    CALL db.index.vector.create(
        'moviePlots_1536', 
        'Movie', 
        'embedding_1536', 
        {metric: 'cosine'}
    )
    """)

# ===============================
# 6️⃣ 测试检索
# ===============================
with driver.session() as session:
    query_vec = embedder.embed_query("Toys coming alive")
    result = session.run(
        """
        CALL db.index.vector.queryNodes('moviePlots_1536', $vec, 5)
        YIELD node, score
        RETURN node.title AS title, node.plot AS plot, score
        """,
        vec=query_vec
    )

    print("Top 5 results:")
    for r in result:
        print(f"{r['title']} (score: {r['score']})")

# ===============================
# 7️⃣ 关闭连接
# ===============================
driver.close()
