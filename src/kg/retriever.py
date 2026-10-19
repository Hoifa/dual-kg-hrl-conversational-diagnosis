# src/kg/retriever.py
from src.kg.connector import neo4j_connector  # 导入刚才写好的单例


class KGRetriever:
    def __init__(self):
        # 不再需要在 __init__ 里建立连接，connector 已经做好了
        self.connector = neo4j_connector

    def close(self):
        # 通常不需要在这里关闭，连接器由主程序统一关闭
        # 但如果非要关，可以调用 self.connector.close()
        pass

    def get_candidates(self, node_id, already_asked_entities=[]):
        """
        全量获取候选实体 (无 Limit)
        """
        query = """
        MATCH (n)
        WHERE $node_id IN n.mapped_node_ids
        AND NOT n.name IN $asked_list
        RETURN n.name AS name, n.label AS label, count{(n)--()} AS degree
        ORDER BY degree DESC
        """

        # 使用 connector 统一执行查询
        parameters = {
            "node_id": node_id,
            "asked_list": already_asked_entities
        }

        results = self.connector.execute_query(query, parameters)

        # 解析结果：execute_query 返回的是字典列表 [{'name': '...', 'degree': ...}, ...]
        candidates = [record['name'] for record in results]

        return candidates

    def get_related_entities(self, current_symptom):
        query = """
        MATCH (n {name: $name})-[r]-(related)
        RETURN related.name AS name, type(r) AS relation
        """
        results = self.connector.execute_query(query, {"name": current_symptom})
        return [record['name'] for record in results]


# ================= 测试代码 =================
if __name__ == "__main__":
    retriever = KGRetriever()

    print("正在测试通过 Connector 检索...")
    candidates = retriever.get_candidates(53, [])

    print(f"检索到 {len(candidates)} 个候选实体")
    print(candidates[:5])  # 打印前5个看看

    # 程序结束时，可以显式关闭连接
    neo4j_connector.close()
