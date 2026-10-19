import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# 尝试从根目录导入配置
try:
    from config import NEO4J_CONFIG
except ImportError:
    # 兼容在子目录下直接运行脚本的情况
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from config import NEO4J_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnector:
    _instance = None

    def __new__(cls):
        """实现单例模式，确保全局只有一个连接器实例"""
        if cls._instance is None:
            cls._instance = super(Neo4jConnector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # 避免单例重复初始化
        if self._initialized:
            return

        self.uri = NEO4J_CONFIG['uri']
        self.user = NEO4J_CONFIG['user']
        self.password = NEO4J_CONFIG['password']
        self.driver = None

        self._connect()
        self._initialized = True

    def _connect(self):
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # 验证连接是否成功
            self.driver.verify_connectivity()
            logger.info(f"✅ 成功连接到 Neo4j 数据库: {self.uri}")
        except ServiceUnavailable as e:
            logger.error(f"❌ 无法连接到 Neo4j，请检查数据库是否启动: {e}")
            raise e
        except Exception as e:
            logger.error(f"❌ Neo4j 连接发生未知错误: {e}")
            raise e

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接已关闭。")

    def execute_query(self, query, parameters=None):
        """
        执行 Cypher 查询的通用方法。

        :param query: Cypher 查询语句
        :param parameters: 查询参数 (字典)
        :return: 结果列表 (List of Dictionaries)
        """
        if not self.driver:
            self._connect()

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                # 将结果转换为字典列表，避免 Session 关闭后无法读取
                # data() 方法会自动将 Neo4j 的 Node/Relationship 对象转为 Python 字典
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"查询执行失败: {e}\nQuery: {query}")
            return []

    def get_driver(self):
        """如果外部需要直接操作 driver (例如高级事务处理)，可调用此方法"""
        return self.driver


# === 实例化导出 ===
# 其他模块直接 import 这个实例，而不是类
neo4j_connector = Neo4jConnector()
