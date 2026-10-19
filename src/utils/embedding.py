import sys
import os
import io
import ollama
# ==========================================
# 1. Windows 中文编码终极修复 (放在所有 import 之前)
# ==========================================
# 强制将标准输出和错误流重定向为 UTF-8，防止打印中文报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ==========================================

from openai import OpenAI
import numpy as np

# 导入根目录下的 config
# 这一步是为了确保能找到 config.py，如果报错说找不到 config，请确保运行目录在项目根目录
try:
    from config import DOCTOR_LLM_CONFIG, EMBEDDING_MODEL
except ImportError:
    # 如果找不到，尝试手动添加路径
    EMBEDDING_MODEL = "bge-m3"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(project_root)
    from config import DOCTOR_LLM_CONFIG, EMBEDDING_MODEL


# class TextEmbedder:
#     def __init__(self):
#         print(f"🔌 初始化 Embedder, 使用模型: {EMBEDDING_MODEL}")
#         print(f"🔗 API Base: {DOCTOR_LLM_CONFIG['base_url']}")
#
#         # 使用 config.py 里的配置
#         self.client = OpenAI(
#             api_key=DOCTOR_LLM_CONFIG['api_key'],
#             base_url=DOCTOR_LLM_CONFIG['base_url']
#         )
#         self.cache = {}
#
#     def get_embedding(self, text: str):
#         """输入文本，返回向量"""
#         # 1. 判空处理
#         if not text or not isinstance(text, str):
#             return [0.0] * 1536  # 假设维度是 1536，阿里 v2 也是 1536
#
#         # 2. 查缓存
#         if text in self.cache:
#             return self.cache[text]
#
#         try:
#             # 去掉换行符，很多 Embedding 模型对换行敏感
#             clean_text = text.replace("\n", " ")
#
#             # 3. 调用 API
#             response = self.client.embeddings.create(
#                 input=[clean_text],
#                 model=EMBEDDING_MODEL  # 使用 config 中配置的正确模型名
#             )
#
#             # 4. 提取向量
#             # 注意：阿里云返回的 object 可能结构略有不同，但兼容 OpenAI 格式
#             vec = response.data[0].embedding
#
#             # 简单验证维度 (防止模型配错)
#             if len(vec) != 1536:
#                 # 阿里 text-embedding-v2 默认是 1536 维
#                 pass
#
#             self.cache[text] = vec
#             return vec
#
#         except Exception as e:
#             # 这里的 print 已经被上面的 sys.stdout 修复了，不会再报 ascii 错误
#             print(f"⚠️ Embedding Error: {e}")
#             print(f"   出错文本片段: {text[:10]}...")
#             # 出错返回零向量，保证程序不崩溃
#             return [0.0] * 1536
#
#
# # 全局单例
# text_embedder = TextEmbedder()
class TextEmbedder:
    def __init__(self, model_name=None):
        # 1. 确定模型名称
        # 如果 config 里有就用 config 的，否则用默认 "bge-m3"
        self.model_name = model_name if model_name else (
            EMBEDDING_MODEL if 'EMBEDDING_MODEL' in globals() else "bge-m3")

        print(f"🔌 初始化 Ollama Embedder...")
        print(f"🧠 使用模型: {self.model_name}")

        # 2. 自动确定维度 (关键！)
        # bge-m3 -> 1024
        # nomic-embed-text -> 768
        # llama3 -> 4096
        if "bge-m3" in self.model_name:
            self.dimension = 1024
        elif "nomic" in self.model_name:
            self.dimension = 768
        elif "llama3" in self.model_name:
            self.dimension = 4096
        else:
            # 其他模型默认按 768 处理，或者你可以先跑一次看长度
            self.dimension = 768
            print(f"⚠️ 警告: 未知模型 {self.model_name}，默认维度设为 768。如果报错请手动修改 embedding.py")

        print(f"📏 向量维度: {self.dimension}")

        # 3. 缓存机制 (RL训练加速神器)
        self.cache = {}

    def get_embedding(self, text: str):
        """输入文本，返回向量"""
        # 判空
        if not text or not isinstance(text, str):
            return [0.0] * self.dimension

        # 查缓存
        if text in self.cache:
            return self.cache[text]

        try:
            # 清洗文本 (Ollama 对换行符通常不敏感，但清洗一下更保险)
            clean_text = text.replace("\n", " ").strip()

            # 如果文本为空，直接返回零向量
            if not clean_text:
                return [0.0] * self.dimension

            # ==========================
            # 🚀 调用本地 Ollama API
            # ==========================
            response = ollama.embeddings(
                model=self.model_name,
                prompt=clean_text
            )

            vec = response["embedding"]

            # 验证维度 (可选)
            if len(vec) != self.dimension:
                # 这种情况很少发生，除非模型自动截断了
                pass

            # 写入缓存
            self.cache[text] = vec
            return vec

        except Exception as e:
            print(f"⚠️ Ollama Embedding Error: {e}")
            print(f"   出错文本: {text[:20]}...")
            # 出错返回零向量，防止训练中断
            return [0.0] * self.dimension


# 全局单例
text_embedder = TextEmbedder()