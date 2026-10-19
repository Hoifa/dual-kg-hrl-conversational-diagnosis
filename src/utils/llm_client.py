# src/utils/llm_client.py
from openai import OpenAI
from config import DOCTOR_LLM_CONFIG, PATIENT_LLM_CONFIG

class LLMService:
    def __init__(self, config):
        """
        初始化一个特定配置的 LLM 客户端
        config: 字典，包含 api_key, base_url, model_name 等
        """
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.7)

    def chat(self, messages: list, system_prompt: str = None) -> str:
        """通用对话方法"""
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM ({self.model_name}): {e}")
            return ""

    def get_embedding(self, text: str):
        """
        获取向量。注意：通常患者 LLM 不需要生成向量，
        只有医生端需要向量来做 '共享底座' 的状态向量。
        """
        text = text.replace("\n", " ")
        # 这里为了统一，建议 Embedding 始终使用效果最好的那个（如 OpenAI），
        # 即使是 Qwen 也可以调 OpenAI 的 Embedding 接口，或者本地起一个 BERT。
        # 这里假设使用 Doctor 的 Embedding 能力。
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-small" # 可以在 config 里配
        )
        return response.data[0].embedding

# --- 实例化两个单例 ---
# 医生 Agent 专用
doctor_llm = LLMService(DOCTOR_LLM_CONFIG)

# 患者 Simulator 专用
patient_llm = LLMService(PATIENT_LLM_CONFIG)
