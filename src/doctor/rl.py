# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import json
import os
import sys
import re
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel  # 🔥 修改点补充: 引入 AutoModel 以防你需要换成 Encoder 架构
import copy
import logging
logger = logging.getLogger(__name__)
import os
import sys

# 定义当前文件的 PROJECT_ROOT
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# ====================================================
# 🟢 1. [Backbone] LLM Embedding
# ====================================================
class PretrainedStateEncoder(nn.Module):
    def __init__(self, model_name=None, freeze=True, device='cpu', args=None):
        super().__init__()
        self.device = device

        # ====================== 通过 --model_path 判断模型类型 ======================
        model_type = "qwen"  # 默认
        actual_model_path = model_name

        if actual_model_path is None and args is not None:
            actual_model_path = getattr(args, 'model_path', None)

        if actual_model_path:
            path_lower = str(actual_model_path).lower()
            if 'glm' in path_lower or 'chatglm' in path_lower:
                model_type = "glm"
            elif 'qwen' in path_lower:
                model_type = "qwen"

        self.model_type = model_type
        print(f"📥 [Backbone] 根据 model_path 识别模型类型: {model_type.upper()}")

        # ====================== 路径处理 ======================
        if model_name is None:
            if model_type == "glm":
                model_path = "/data/pretrained_models/GLM-4-9B-Chat"
                hf_fallback = "THUDM/GLM-4-9B-Chat"
            else:  # qwen
                model_path = "/data/pretrained_models/Qwen2.5-7B-Instruct"
                hf_fallback = "Qwen/Qwen2.5-7B-Instruct"

            if os.path.exists(model_path):
                model_name = model_path
            else:
                model_name = hf_fallback
                print(f"⚠️ 未找到本地路径，使用 HF 兜底: {hf_fallback}")

        # ====================== 公共 Tokenizer 处理 ======================
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # ====================== 模型专属加载 ======================
        try:
            if model_type == "glm":
                self._load_glm(model_name)
            else:
                self._load_qwen(model_name)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

        if freeze:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()

        self.output_dim = self.llm.config.hidden_size
        self.embedding_cache = {}

    def _load_qwen(self, model_name):
        print(f"📥 [Backbone] 加载 Qwen2.5: {model_name}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

    def _load_glm(self, model_name):
        print(f"📥 [Backbone] 加载 GLM: {model_name}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        # 你原本的设置可以保留
        self.llm.config.use_cache = False

        # ==================== 注入安全垫片（关键：解决之前的崩溃报错） ====================

        # 1. 修复 generate 时的 attention_mask 和参数警告
        original_generate = self.llm.generate

        def safe_glm_generate(*args, **kwargs):
            if len(args) > 0:
                input_ids = args[0]
            else:
                input_ids = kwargs.get("input_ids")
            if input_ids is not None and "attention_mask" not in kwargs:
                kwargs["attention_mask"] = torch.ones_like(input_ids).to(input_ids.device)
            if kwargs.get("do_sample") is False:
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
            return original_generate(*args, **kwargs)

        self.llm.generate = safe_glm_generate

        # 2. 修复 forward 时的 AttributeError: 'NoneType' object has no attribute 'shape'
        original_forward = self.llm.forward

        def safe_glm_forward(*args, **kwargs):
            past_key_values = kwargs.get("past_key_values", None)
            if past_key_values is not None:
                try:
                    # 如果是 transformers 传进来的空缓存结构，强制设为 None
                    if hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
                        kwargs["past_key_values"] = None
                    elif isinstance(past_key_values, (tuple, list)):
                        if len(past_key_values) == 0 or past_key_values[0] is None or past_key_values[0][0] is None:
                            kwargs["past_key_values"] = None
                except Exception:
                    pass
            return original_forward(*args, **kwargs)

        self.llm.forward = safe_glm_forward

    def forward(self, text_list):
        # ⚠️ 直接作为透传接口，不做任何模板包装，避免套娃！
        return self._get_vectors(text_list)

    # _get_vectors 方法保持你当前版本（带 batching 的修复版）即可
    def get_candidate_embeddings(self, candidates_list):
        if not candidates_list:
            return torch.empty(0, self.output_dim).to(self.device)
        vectors = []
        uncached_texts = []
        uncached_indices = []

        for idx, text in enumerate(candidates_list):
            # 🔥 在这里为知识图谱的实体也套上通用的对话模板
            msg = [{"role": "user", "content": f"医学实体特征提取：{text}"}]
            fmt_txt = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )

            # 使用格式化后的文本作为 Cache Key
            if fmt_txt in self.embedding_cache:
                vectors.append(self.embedding_cache[fmt_txt])
            else:
                vectors.append(None)
                uncached_texts.append(fmt_txt)
                uncached_indices.append(idx)

        if uncached_texts:
            new_vectors = self._get_vectors(uncached_texts)
            for i, vec_tensor in enumerate(new_vectors):
                self.embedding_cache[uncached_texts[i]] = vec_tensor
                vectors[uncached_indices[i]] = vec_tensor

        return torch.stack(vectors)

    # def _get_vectors(self, text_list):
    #     encoded_input = self.tokenizer(
    #         text_list, padding=True, truncation=True, max_length=1024, return_tensors='pt'
    #     ).to(self.device)
    #     with torch.no_grad():
    #         outputs = self.llm(**encoded_input, output_hidden_states=True)
    #     # 取最后一个 token 的 hidden state 作为句子嵌入
    #     # 🔥 修改点 4 (注意): 如果你换成了 BERT 类的模型，你需要将 outputs.hidden_states[-1][:, -1, :]
    #     # 修改为取 [CLS] token，即 outputs.last_hidden_state[:, 0, :]。若依然是 Qwen/Llama 等因果模型则无需修改。
    #     return outputs.hidden_states[-1][:, -1, :].float()


    def _get_vectors(self, text_list):
        """
        🔥 修复版：防止知识图谱实体过多导致显存爆炸
        """
        if not text_list:
            return torch.empty(0, self.output_dim).to(self.device)

        # 1. 设置合适的分批大小，16 或 32 是非常安全的
        BATCH_SIZE = 16
        all_vectors = []

        # 2. 显式开启 no_grad，这是大模型提取特征免爆显存的金牌护身符
        with torch.no_grad():
            for i in range(0, len(text_list), BATCH_SIZE):
                batch_texts = text_list[i: i + BATCH_SIZE]

                # Tokenize 当前批次
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,  # 限制最大长度，防止异常长文本
                    return_tensors="pt"
                ).to(self.device)

                # 模型前向传播
                outputs = self.llm(**encoded_input, output_hidden_states=True)

                # 3. 提取特征向量 (请对照你原代码第 140 行左右的提取逻辑，保持一致)
                # 华佗/百川架构通常从 hidden_states[-1] 拿最后一层
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    last_hidden_state = outputs.hidden_states[-1]
                else:
                    last_hidden_state = outputs.last_hidden_state

                # 因为前面设置了 padding_side='left'，所以最后一项 [:, -1, :] 永远是真实文本的结尾向量
                batch_vectors = last_hidden_state[:, -1, :]

                # 先移到 CPU 寄存，彻底断绝 GPU 积压
                all_vectors.append(batch_vectors.cpu())

                # 4. 手动清理当前循环的显存变量（双重保险）
                del encoded_input, outputs

        # 将所有批次的向量拼接，并统一送回 GPU 设备
        return torch.cat(all_vectors, dim=0).to(device=self.device, dtype=torch.float32)
# ====================================================
# 🟢 2. [KGRetriever] (保持不变)
# ====================================================
class KGRetriever:
    def __init__(self, kg_path, num_hla_nodes):
        self.kg_path = kg_path
        self.num_hla_nodes = num_hla_nodes
        self.hla_to_candidates = {i: set() for i in range(num_hla_nodes)}
        self._load_and_build_mapping()

    def _load_and_build_mapping(self):
        if not os.path.exists(self.kg_path):
            print(f"⚠️ [KGRetriever] 未找到 KG 文件: {self.kg_path}")
            return
        try:
            df = pd.read_csv(self.kg_path)
            for _, row in df.iterrows():
                name = str(row['name']).strip()
                ids_raw = row['mapped_node_ids']
                try:
                    if isinstance(ids_raw, str):
                        target_ids = ast.literal_eval(ids_raw)
                    elif isinstance(ids_raw, (int, float)):
                        target_ids = [int(ids_raw)]
                    else:
                        target_ids = []
                except:
                    target_ids = []

                if isinstance(target_ids, list):
                    for nid in target_ids:
                        nid = int(nid)
                        if nid < self.num_hla_nodes:
                            self.hla_to_candidates[nid].add(name)
        except Exception as e:
            print(f"⚠️ [KGRetriever] 加载 CSV 失败: {e}")

    def get_all_candidates(self, hla_id):
        return self.hla_to_candidates.get(hla_id, set())

    def get_unasked_candidates(self, hla_id, asked_history_set):
        all_cands = self.hla_to_candidates.get(hla_id, set())
        valid_cands = list(all_cands - asked_history_set)
        return valid_cands


# ====================================================
# 🟢 3. [Diagnosis Head] (MLP Version) - 智能自适应版
# ====================================================
class DiagnosisHead(nn.Module):
    def __init__(self, state_dim, num_diseases):
        super().__init__()

        # 🔥 核心修改：统一使用 [1024 -> 512] 结构，头尾维度由变量自动适应
        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),  # 自动适应 3584 (Qwen) 或 4096 (GLM)
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_diseases)  # 自动适应分类数 (如 7 或 16)
        )

        model_type = "GLM" if state_dim == 4096 else "Qwen"
        print(f"🧠 [DiagnosisHead] 检测到输入维度 {state_dim} ({model_type}架构)，已统一匹配 [1024 -> 512] 诊断头！")

    def forward(self, state_embedding):
        return self.net(state_embedding)

    def load_pretrained_weights(self, pth_path):
        if pth_path and os.path.exists(pth_path):
            device = next(self.parameters()).device
            try:
                state_dict = torch.load(pth_path, map_location=device)

                # 兼容格式：如果是保存的完整的训练状态 dict（包含 model_state_dict）
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

                new_state_dict = {}
                for k, v in state_dict.items():
                    # 1. 强力去除各种可能存在的前缀
                    clean_k = k.replace('diag_head.', '').replace('module.', '')

                    # 2. 核心：如果发现权重里包含 classifier.xxx（比如从静态评估导出的结构）
                    # 自动将其映射到强化学习需要的 net.xxx
                    clean_k = clean_k.replace('classifier.', 'net.')

                    # 3. 如果这个 key 属于大模型（比如 base_model），直接过滤扔掉，我们这里只要分类头
                    if clean_k.startswith('base_model') or 'transformer' in clean_k:
                        continue

                    new_state_dict[clean_k] = v

                # 4. 动态检查：检查我们剥离出来的 key 是不是叫 0.weight 这种 nn.Sequential 默认名
                # 如果是，强行加上 "net." 前缀以匹配我们的 self.net
                final_state_dict = {}
                for k, v in new_state_dict.items():
                    if k.startswith(('0.', '1.', '2.', '3.', '4.', '5.', '6.')):
                        final_state_dict[f"net.{k}"] = v
                    else:
                        final_state_dict[k] = v

                if len(final_state_dict) == 0:
                    print(f"⚠️ [Warning] 在权重文件中未找到任何有效的分类头参数，将保持随机初始化！")
                    return

                # 5. 换成比较安全的非严格加载，但打印匹配详情，防止闷声犯大错
                print(f"📥 [DiagnosisHead] 正在剥离并加载分类头参数，共识别到 {len(final_state_dict)} 个有效张量...")
                missing_keys, unexpected_keys = self.load_state_dict(final_state_dict, strict=False)

                if missing_keys:
                    print(f"⚠️ [加载提示] 分类头缺失的键(将保持随机): {missing_keys}")
                else:
                    print(f"✅ [DiagnosisHead] 恭喜！诊断头预训练权重已实现 100% 完美对齐和无缝注入！")

            except Exception as e:
                print(f"❌ [致命错误] 诊断头权重加载失败: {e}")
                raise e
# ====================================================
# 🟢 4. [DoctorAgent] 主智能体
# ====================================================

class DoctorAgent(nn.Module):
    def __init__(self, data_dir=None, model_weight_path=None, device=None, args=None,
                 label_path=None, csv_path=None, kg_entity_path=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args is not None:
            self.dataset_name = str(getattr(args, 'data', 'dlbcl'))
        else:
            self.dataset_name = 'dlbcl'
        print(f"🌟 当前初始化的数据集路径为: {self.dataset_name}")

        # ==========================================
        # 📂 1. 路径重构与自动推断 (修复 NoneType 报错)
        # ==========================================
        # 优先使用显式传入的参数
        self.csv_path = csv_path
        self.label_path = label_path
        self.kg_entity_path = kg_entity_path

        # 如果没有传入 data_dir，则尝试从 args 中获取
        if data_dir is None and args is not None:
            data_dir = getattr(args, 'data_dir', None)

        # 🔥 核心修正：如果路径还是空的，基于已有的 csv_path 或 data_dir 进行推断
        if not self.csv_path and data_dir:
            self.csv_path = os.path.join(data_dir, "wen.csv")

        if not self.kg_entity_path:
            if data_dir:
                self.kg_entity_path = os.path.join(data_dir, "disease_kg_entity.csv")
            elif self.csv_path:
                # 如果没有 data_dir，假设 KG 文件和 wen.csv 在同一个目录下
                self.kg_entity_path = os.path.join(os.path.dirname(self.csv_path), "disease_kg_entity.csv")

        if not self.label_path:
            if data_dir:
                self.label_path = os.path.join(data_dir, "disease_labels.json")
            elif self.csv_path:
                self.label_path = os.path.join(os.path.dirname(self.csv_path), "disease_labels.json")

        # 最终校验：确保关键路径不再是 None
        if self.csv_path is None or self.kg_entity_path is None:
            raise ValueError(f"🚨 [DoctorAgent] 初始化失败！无法确定数据路径。\n"
                             f"csv_path: {self.csv_path}\n"
                             f"kg_path: {self.kg_entity_path}")

        # ==========================================
        # ⚖️ 2. 权重路径处理 —— 仅通过完整 --model_path 判断（修复 UnboundLocalError）
        # ==========================================
        if model_weight_path is None:
            SAVE_DIR = os.path.join(PROJECT_ROOT, "src", "model", "checkpoints")

            # 优先从 args 获取完整路径
            bp = getattr(args, 'model_path', None) if args is not None else None

            # 🔥 关键修复：无论是否进入 if，都必须初始化
            model_folder_name = "Qwen2.5"  # 默认值

            if bp:
                # 从完整路径 basename 提取模型类型
                path_basename = os.path.basename(str(bp).rstrip('/'))
                if 'glm' in path_basename.lower() or 'chatglm' in path_basename.lower():
                    model_folder_name = "GLM-4-9B-Chat"
                elif 'qwen' in path_basename.lower():
                    model_folder_name = "Qwen2.5"

            model_weight_path = os.path.join(
                SAVE_DIR,
                f"diag_head_{model_folder_name}_{self.dataset_name}_best.pth"
            )

        if 'model_folder_name' not in locals():
            model_folder_name = "Qwen2.5"  # 兜底
            import re
            # 将 filename 替换为该函数/方法中实际持有的权重路径变量（通常是 model_weight_path）
            match = re.search(r"diag_head_(.*?)_[^_]+_best\.pth", model_weight_path, re.IGNORECASE)
            if 'glm' in str(model_weight_path).lower() or 'chatglm' in str(model_weight_path).lower():
                model_folder_name = "GLM-4-9B-Chat"

        self.model_weight_path = model_weight_path
        print(f"📌 使用的诊断头权重路径: {self.model_weight_path} （model: {model_folder_name}）")
        # 最终校验
        if not os.path.exists(self.model_weight_path):
            print(f"⚠️ 警告：权重文件不存在: {self.model_weight_path}")
        print(f"📦 [DoctorAgent] 正在加载资源...")
        print(f"   - CSV: {self.csv_path}")
        print(f"   - KG:  {self.kg_entity_path}")

        # --- 加载资源 (注意：这里一定要用 self.xxx 变量) ---
        try:
            self.node_df = pd.read_csv(self.csv_path, encoding='utf-8')
        except:
            self.node_df = pd.read_csv(self.csv_path, encoding='gbk')

        self.num_nodes = int(self.node_df['node_id'].max()) + 1
        self.DIAGNOSIS_NODE_ID = 93

        self.backbone = PretrainedStateEncoder(freeze=True, device=self.device,args=args)
        self.state_dim = self.backbone.output_dim

        # 此时 self.kg_entity_path 已经是字符串，不再是 None
        self.retriever = KGRetriever(self.kg_entity_path, self.num_nodes)

        with open(self.label_path, 'r', encoding='utf-8') as f:
            raw_map = json.load(f)
        self.disease_map = self._auto_parse_json(raw_map)
        self.idx_to_disease_name = {v: k for k, v in self.disease_map.items()}
        self.num_diseases = len(self.disease_map)

        if os.path.exists(self.kg_entity_path):
            df_ent = pd.read_csv(self.kg_entity_path)
            self.entity_list = df_ent['name'].astype(str).str.strip().drop_duplicates().tolist()
            self.entity2id = {name: i for i, name in enumerate(self.entity_list)}
            self.lla_dim = len(self.entity_list)
            with torch.no_grad():
                self.all_entity_embeddings = self.backbone.get_candidate_embeddings(self.entity_list)
        else:
            self.entity2id, self.lla_dim = {}, 0

        # --- 网络模块初始化 ---
        self.hla_net = nn.Sequential(nn.Linear(self.state_dim, 256), nn.ReLU(), nn.Linear(256, self.num_nodes)).to(
            self.device)
        self.target_hla_net = copy.deepcopy(self.hla_net)
        for param in self.target_hla_net.parameters():
            param.requires_grad = False

        self.lla_actor = nn.Bilinear(self.state_dim, self.state_dim, 1).to(self.device)
        self.lla_critic = nn.Sequential(nn.Linear(self.state_dim, 256), nn.ReLU(), nn.Linear(256, 1)).to(
            self.device)
        self.diag_head = DiagnosisHead(state_dim=self.state_dim, num_diseases=self.num_diseases).to(self.device)

        # 加载诊断头权重
        if model_weight_path and os.path.exists(model_weight_path):
            self.diag_head.load_pretrained_weights(model_weight_path)
        else:
            print(f"⚠️ [Warning] 未找到权重文件: {model_weight_path}")

        self.asked_node_ids = set()
        self.asked_entities = set()  # 👈 必须在这里先声明一下
        self.struct_memory = {
            "主诉": "",
            "阳性": set(),
            "阴性": set()
        }
        self.reset()
        # ====================== 新增：HLA动作模式开关 ======================
        # 支持三种模式：'pure_inquiry'(0-47)、'inquiry_plus_physical'(0-58)、'full'(原逻辑)
        self.hla_mode = 'full'   # ← 这里切换模式
        print(f"🔧 [DoctorAgent] 已启用 HLA 模式: {self.hla_mode} "
              f"(纯问诊=0-47 | 问诊+体查=0-58)")
        # 在 __init__ 方法末尾添加：
        self.epsilon = getattr(args, 'epsilon', 0.1)  # 默认给一个 0.1 的探索率


    def _auto_parse_json(self, raw_map):
        if not raw_map: return {}
        first_k = list(raw_map.keys())[0]
        if str(first_k).isdigit():
            return {str(v).strip(): int(k) for k, v in raw_map.items()}
        return {str(k).strip(): int(v) for k, v in raw_map.items()}

    def reset(self):
        self.asked_entities.clear()
        self.struct_memory = {
            "主诉": "",
            "阳性": set(),
            "阴性": set()
        }
        self.hla_history = []
        self.asked_entities = set()
        self.asked_node_ids = set()

    # ====================================================
    # 🔥 核心修改: 结构化记忆更新
    # ====================================================
    import json
    import re

    def parse_observation(self, natural_text, turn_idx, prev_memory=None, doctor_question=""):
        if not natural_text or len(natural_text.strip()) < 2:
            return {"主诉": "", "阳性": [], "阴性": []}

        # 🛑 核心修复 1：Python 物理拦截“未做检查”
        if "未见" in natural_text and ("检查" in natural_text or "该项" in natural_text) and "异常" not in natural_text:
            return {"主诉": "", "阳性": [], "阴性": []}
        if any(kw in natural_text for kw in ["未做", "未查", "无此检查", "拒查"]):
            return {"主诉": "", "阳性": [], "阴性": []}

        # 🧠 核心修复 2：Prompt 强引导“检查前缀”区分阴阳性
        prompt = f"""你是一名医学数据提取专家。请从对话中提取关键信息。 
            当前对话轮次: {turn_idx} (如果是0轮，请特别留意提取'主诉') 
            医生提问/动作: "{doctor_question}" 
            患者/系统回复: "{natural_text}" 

            请提取以下字段并以 JSON 格式输出： 
            - 主诉: 患者最主要的症状描述 
            - 阳性: 确认存在的症状、**异常**的体检或化验结果。(🔥极其重要：如果是异常的辅助检查或体征，请提取名称作为前缀，例如“CT平扫检查示:双肺纹理增多”、“骨髓穿刺及活检示:骨髓增生活跃”) 
            - 阴性: 明确否认的症状、**正常**的检查结果。(🔥极其重要：如果是正常的检查结果，同样需要加上前缀并放入此处！例如“超声检查示:未见异常”、“皮肤黏膜与淋巴结检查示:未见明显异常”) 

            注意：只需输出 JSON，严禁任何解释。 
            格式示例：{{"主诉": "...", "阳性": ["CT平扫检查示:右下肺小结节影"], "阴性": ["关节活动受限", "超声检查示:未见异常"]}}"""

        input_ids = self.backbone.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.backbone.llm.generate(
                input_ids,
                max_new_tokens=256,
                # +++ 【优化 1】当 do_sample=False 时移除 temperature 参数，防止 Transformers 报 UserWarning 警告 +++
                do_sample=False
                # +++
            )
            response = self.backbone.tokenizer.decode(
                output_ids[0][len(input_ids[0]):],
                skip_special_tokens=True
            )

            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    pos = list(set(data.get("阳性", []) if isinstance(data.get("阳性"), list) else []))
                    neg = list(set(data.get("阴性", []) if isinstance(data.get("阴性"), list) else []))

                    # 🛡️ 过滤废话与漏网之鱼
                    invalid_exact_matches = ["无", "没有", "未见", "不相关", "未做", "无记录", "否", "正常", "未见异常",
                                             "未见明显异常"]
                    filtered_neg = []
                    for x in neg:
                        # +++ 【优化 2】增加 isinstance(x, str) 防御性断言，防止大模型抽风输出非文本类型导致代码挂掉 +++
                        if not isinstance(x, str):
                            continue
                        # +++

                        # 🚨 双保险：如果大模型在阴性里输出了 "未见xxx检查" 这种没做检查的废话，直接扔掉！
                        if "未见" in x and "检查" in x and "异常" not in x:
                            continue
                        if len(x.strip()) >= 1 and x.strip() not in invalid_exact_matches:
                            clean_x = x.replace("没有", "").replace("无", "").strip()
                            if clean_x:
                                filtered_neg.append(clean_x)

                    chief_complaint = data.get("主诉", "")
                    if not isinstance(chief_complaint, str):
                        chief_complaint = str(chief_complaint)

                    if turn_idx > 0 and (not chief_complaint.strip()) and prev_memory:
                        chief_complaint = prev_memory.get("主诉", "")

                    return {
                        "主诉": chief_complaint,
                        "阳性": pos,
                        "阴性": filtered_neg
                    }
            except Exception as e:
                # +++ 【优化 3】动态化报错日志，清晰利用 self.backbone.model_type 打印出当前是哪个模型解析失败 +++
                model_tag = getattr(self.backbone, 'model_type', 'LLM').upper()
                print(f"⚠️ [{model_tag} 提取结构化信息失败]: {e} | 原始回复: {response}")
                # +++

            return {"主诉": "", "阳性": [], "阴性": []}
    def update_struct_memory(self, extracted_dict, turn_idx):
        """
        将本轮提取到的信息合入全局记忆
        """
        # 1. 主诉处理：仅在首轮或当前记忆为空时更新
        new_cc = str(extracted_dict.get("主诉", "")).strip()
        if new_cc and (not self.struct_memory["主诉"] or turn_idx == 0):
            self.struct_memory["主诉"] = new_cc

        # 2. 阳性累积
        for item in extracted_dict.get("阳性", []):
            item = item.strip()
            if item:
                self.struct_memory["阳性"].add(item)
                # 冲突纠正：如果现在说有了，就从阴性里删掉
                if item in self.struct_memory["阴性"]:
                    self.struct_memory["阴性"].remove(item)

        # 3. 阴性累积
        for item in extracted_dict.get("阴性", []):
            item = item.strip()
            if item:
                self.struct_memory["阴性"].add(item)
                # 冲突纠正
                if item in self.struct_memory["阳性"]:
                    self.struct_memory["阳性"].remove(item)

    def _update_target_network(self, tau=0.005):
        """软更新 target_hla_net"""
        if not hasattr(self, 'target_hla_net'):
            return
        for param, target_param in zip(
            self.hla_net.parameters(),
            self.target_hla_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    # def _get_memory_text_representation(self):
    #     """
    #     🔥 序列化记忆 -> Standard Text
    #     格式: "主诉：xxx。阳性症状：xxx，xxx。阴性症状：xxx，xxx。"
    #     严格对齐训练数据的格式。
    #     """
    #     # print(f"DEBUG MEMORY: {self.struct_memory}")
    #     cc = self.struct_memory["主诉"] if self.struct_memory["主诉"] else "未述"
    #
    #     # 将集合转换为逗号分隔的字符串
    #     pos_list = list(self.struct_memory["阳性"])
    #     pos_str = "，".join(pos_list) if pos_list else "暂无"
    #
    #     neg_list = list(self.struct_memory["阴性"])
    #     neg_str = "，".join(neg_list) if neg_list else "暂无"
    #
    #     # 构造最终文本 (注意中文标点)
    #     text = f"主诉：{cc}。阳性症状：{pos_str}。阴性症状：{neg_str}。"
    #     return text
    #
    # def get_structured_state_for_diagnosis(self):
    #     """获取用于诊断和 RL 的 State Embedding"""
    #     text = self._get_memory_text_representation()
    #     # backbone 返回 [1, Hidden_Dim]
    #     return self.backbone([text]).detach()
    def _get_memory_text_representation(self):
        """
        🔥 序列化记忆 -> Standard Text
        保持纯净的自然语言格式，不带任何模型特定的 Tag
        """
        cc = self.struct_memory["主诉"] if self.struct_memory["主诉"] else "未述"

        # 将集合转换为逗号分隔的字符串
        pos_list = list(self.struct_memory["阳性"])
        pos_str = ", ".join(pos_list) if pos_list else "无"

        neg_list = list(self.struct_memory["阴性"])
        neg_str = ", ".join(neg_list) if neg_list else "无"

        # 严格对齐诊断头训练时的基础内容格式
        text = f"医学诊断决策分析：\n[主诉情况]: {cc}\n[阳性体征/检查]: {pos_str}\n[阴性排除项]: {neg_str}\n请基于证据给出临床诊断："
        return text

    def get_structured_state_for_diagnosis(self):
        # 1. 拿到纯粹的病历文本
        raw_text = self._get_memory_text_representation()

        # 2. 直接喂给 backbone，不要套用任何 messages 或 chat_template！
        # 保持和离线预训练诊断头时一模一样的输入分布
        return self.backbone([raw_text]).detach()

    def get_state_embedding(self, text=None):

        return self.get_structured_state_for_diagnosis()

    # def get_hla_mask(self, current_turn=None, max_turns=None, gender=None, age=None):
    #     """🚀 调度员：根据当前初始化的 dataset_name 分发给对应的专科临床路径"""
    #     if self.dataset_name in ["iiyi", "mimiciv", "mimic"]:
    #         return self._get_mask_pneumonia(current_turn, max_turns, gender, age)
    #     else:
    #         # 默认走 DLBCL 逻辑
    #         return self._get_mask_dlbcl(current_turn, max_turns, gender, age)
    def get_hla_mask(self, current_turn=None, max_turns=None, gender=None, age=None):
        # print(f"🚨 当前 dataset_name = {self.dataset_name}")
        # print(f"🚨 当前 hla_mode = {self.hla_mode}")
        """🚀 调度员：根据 hla_mode 分发不同限制策略"""
        if self.hla_mode == 'pure_inquiry':
            return self._get_mask_pure_inquiry(current_turn, max_turns, gender, age)
        elif self.hla_mode == 'inquiry_plus_physical':
            return self._get_mask_inquiry_plus_physical(current_turn, max_turns, gender, age)
        # ==========================================================
        # 兜底：走原有专科逻辑
        # ==========================================================
        # 🟢 修复 1：使用 .lower() 抹平大小写差异，防止 "IIYI" 误入歧途
        if str(self.dataset_name).lower() in ["iiyi", "mimiciv", "mimic"]:
            # print("✅ 进入 pneumonia mask")
            # 🟢 修复 2：将上层收到的参数原封不动传进去，别用省略号 `...`
            return self._get_mask_pneumonia(current_turn, max_turns, gender, age)

        else:
            # print("❌ 进入 dlbcl mask")
            # 🟢 修复 3：同上，将参数正确传递给 dlbcl 掩码函数
            return self._get_mask_dlbcl(current_turn, max_turns, gender, age)


    def _get_mask_dlbcl(self, current_turn=None, max_turns=None, gender=None, age=None):
        """
        获取当前回合可用动作的 Mask (1.0 表示可用，0.0 表示不可用)
        优化版：平衡临床路径与探索激励
        """
        mask = torch.zeros(self.num_nodes).to(self.device)

        # ==========================================
        # 1. 定义节点群组（保持不变）
        # ==========================================
        LAB_NODES = set(range(59, 78))  # 实验室检查
        IMAGING_NODES = set(range(78, 88))  # 影像/器械检查
        INVASIVE_NODES = {88, 89, 90, 91, 92}  # 有创/病理
        DIAG_NODE = 93

        # ==========================================
        # 2. 状态检查：**放宽**前置条件（核心修改）
        # ==========================================
        done_basic_lab = any(node in self.asked_node_ids for node in [59, 60, 61, 71, 72])
        done_liver_func = (62 in self.asked_node_ids)
        # 🔥 修改1：降低门槛，只需基础化验或肝功之一即可视为done_labs
        done_labs = done_basic_lab or done_liver_func  # 原 and → or

        done_imaging = any(node in self.asked_node_ids for node in IMAGING_NODES)
        done_pathology = any(node in self.asked_node_ids for node in INVASIVE_NODES)

        # ==========================================
        # 3. 基础过滤逻辑（保持你的噪声黑名单）
        # ==========================================
        DLBCL_NOISE_NODES = [10, 11, 12, 49, 51, 52, 53, 54, 55, 56, 57, 58,
                             60, 61, 64, 65, 66, 67, 68, 70, 73, 74, 77, 78, 86, 87]

        for i in range(self.num_nodes):
            if i == DIAG_NODE:
                continue
            if i in DLBCL_NOISE_NODES:
                continue
            if (48 <= i <= 92) and i in self.asked_node_ids:
                continue

            unasked = self.retriever.get_unasked_candidates(i, self.asked_entities)
            if len(unasked) > 0:
                mask[i] = 1.0

        # ==========================================
        # 4. 强制临床路径约束（**适当放松**）
        # ==========================================
        # A. 有创检查封锁条件放宽：只需完成化验或影像之一即可尝试有创
        if not (done_labs or done_imaging):  # 原 and → or
            for inv_node in INVASIVE_NODES:
                if inv_node < self.num_nodes:
                    mask[inv_node] = 0.0

        # B. 诊断熔断：**训练早期**降低要求（允许探索）
        if current_turn is not None and current_turn < 6:  # 前6轮放宽
            mask[DIAG_NODE] = 1.0 if done_labs or done_imaging else 0.0
        else:
            if not (done_labs and done_imaging and done_pathology):
                mask[DIAG_NODE] = 0.0
            else:
                mask[DIAG_NODE] = 1.0

        # ==========================================
        # 5. 阶段性解禁（**关键修改**）
        # ==========================================
        if current_turn is not None:
            if current_turn < 2:  # 原 < 2 → < 1，只强制第0轮问诊
                mask[48:93] = 0.0
            elif current_turn < 4:  # 第1-3轮逐步解禁检查
                # 允许实验室和影像
                for node in list(LAB_NODES | IMAGING_NODES):
                    if node < self.num_nodes and node not in self.asked_node_ids:
                        mask[node] = 1.0

        # ==========================================
        # 6. 健壮性兜底 + 主动推动检查（增强版）
        # ==========================================
        if mask.sum() == 0 or (current_turn is not None and current_turn >= 2):
            available_labs = [i for i in LAB_NODES if i not in self.asked_node_ids and i not in DLBCL_NOISE_NODES]
            available_imaging = [i for i in IMAGING_NODES if
                                 i not in self.asked_node_ids and i not in DLBCL_NOISE_NODES]
            available_pathology = [i for i in INVASIVE_NODES if i not in self.asked_node_ids]
            available_inquiry = [i for i in range(48) if i not in self.asked_node_ids]

            if not done_labs and available_labs:
                # 优先强制关键检查
                forced = [i for i in [59, 62] if i not in self.asked_node_ids]  # 血常规 + 肝功
                target_labs = forced if forced else available_labs[:3]
                if target_labs:  # 🔥 修复：安全判定，直接传 List
                    mask[target_labs] = 1.0
            elif not done_imaging and available_imaging:
                target_imaging = available_imaging[:3]
                if target_imaging:  # 🔥 修复
                    mask[target_imaging] = 1.0
            elif not done_pathology and available_pathology:
                if available_pathology:  # 🔥 修复
                    mask[available_pathology] = 1.0
            elif available_inquiry:
                target_inquiry = available_inquiry[:2]
                if target_inquiry:  # 🔥 修复
                    mask[target_inquiry] = 1.0
            else:
                mask[DIAG_NODE] = 1.0

        # **新增调试日志**（训练时观察Mask行为）
        if current_turn is not None and current_turn % 3 == 0:
            lab_avail = mask[list(LAB_NODES)].sum().item()
            img_avail = mask[list(IMAGING_NODES)].sum().item()
            inv_avail = mask[list(INVASIVE_NODES)].sum().item()

            # 🔥 安全调用日志
            # try:
            #     logger.info(f"[Mask Turn {current_turn}] Labs: {lab_avail:.0f} | "
            #                 f"Imaging: {img_avail:.0f} | Invasive: {inv_avail:.0f} | "
            #                 f"Done_labs: {done_labs}")
            # except NameError:
            #     print(f"[Mask Turn {current_turn}] Labs: {lab_avail:.0f} | "
            #           f"Imaging: {img_avail:.0f} | Invasive: {inv_avail:.0f} | "
            #           f"Done_labs: {done_labs}")
        return mask

    # def _get_mask_pneumonia(self, current_turn, max_turns, gender, age):
    #     """
    #     🫁 呼吸/肺炎专科路径 (放宽优化版)
    #     """
    #     # print("🫁 PNEUMONIA MASK")
    #     mask = torch.zeros(self.num_nodes).to(self.device)
    #
    #     LAB_NODES = set(range(59, 78))
    #     IMAGING_NODES = set(range(78, 88))
    #     DIAG_NODE = 93
    #
    #     done_labs = (59 in self.asked_node_ids) or (71 in self.asked_node_ids)
    #     done_chest_imaging = any(node in self.asked_node_ids for node in [79, 81])
    #
    #     # 基础过滤（噪声绝对屏蔽）
    #     PNEUMONIA_NOISE = [51, 52, 54, 55, 56, 57, 58, 72, 73, 74, 76, 82, 84, 85, 88, 89, 90, 91, 92]
    #
    #     for i in range(self.num_nodes):
    #         if i == DIAG_NODE or i in PNEUMONIA_NOISE:
    #             continue
    #         if (48 <= i <= 92) and i in self.asked_node_ids:
    #             continue
    #
    #         unasked = self.retriever.get_unasked_candidates(i, self.asked_entities)
    #         if len(unasked) > 0:
    #             mask[i] = 1.0
    #
    #     # ==========================================
    #     # 4. 强制临床路径约束（放松诊断锁）
    #     # ==========================================
    #     # 训练早期（前6轮）允许探索性诊断；后期必须满足两项检查
    #     if current_turn is not None and current_turn < 6:
    #         mask[DIAG_NODE] = 1.0 if (done_labs or done_chest_imaging) else 0.0  # 早期 done 任意一个即可探索
    #     else:
    #         mask[DIAG_NODE] = 1.0 if (done_labs and done_chest_imaging) else 0.0
    #
    #     # ==========================================
    #     # 5. 阶段性解禁
    #     # ==========================================
    #     if current_turn is not None:
    #         if current_turn < 2:  # 🔴 只强制第 0 轮纯问诊，第 1 轮起允许检查，加快节奏
    #             mask[48:93] = 0.0
    #             mask[DIAG_NODE] = 0.0
    #
    #     # ==========================================
    #     # 6. 健壮性兜底与智能引导（去除硬性收紧）
    #     # ==========================================
    #     if mask.sum() == 0 or (current_turn is not None and current_turn >= 2):
    #         available_labs = [i for i in LAB_NODES if i not in self.asked_node_ids and i not in PNEUMONIA_NOISE]
    #         available_imaging = [i for i in IMAGING_NODES if i not in self.asked_node_ids and i not in PNEUMONIA_NOISE]
    #
    #         # 如果缺失核心检查，强推它们（亮起它们），但【绝不关闭】问诊大门！
    #         if not done_labs and available_labs:
    #             forced_labs = [i for i in [59, 71] if i not in self.asked_node_ids]
    #             target_labs = forced_labs if forced_labs else available_labs[:3]
    #             if target_labs:  # 加个判空，防止 list 为空时报错
    #                 mask[target_labs] = 1.0
    #         if not done_chest_imaging and available_imaging:
    #             forced_imaging = [i for i in [79, 81] if i not in self.asked_node_ids]
    #             target_imaging = forced_imaging if forced_imaging else available_imaging[:3]
    #             if target_imaging:
    #                 mask[target_imaging] = 1.0
    #         # 💡 核心修改：当证据齐备时，允许诊断，同时【保留】问诊和体查，让模型自己学会在信息完美时收手
    #         if done_labs and done_chest_imaging:
    #             mask[DIAG_NODE] = 1.0
    #             # 不再执行 mask[0:59] = 0.0！！！
    #
    #     return mask
    def _get_mask_pneumonia(self, current_turn, max_turns, gender, age):
        """
        🫁 呼吸/肺炎专科路径 (铁血临床路径版)
        """
        mask = torch.zeros(self.num_nodes).to(self.device)

        LAB_NODES = set(range(59, 78))
        IMAGING_NODES = set(range(78, 88))
        DIAG_NODE = 93

        done_labs = (59 in self.asked_node_ids) or (71 in self.asked_node_ids)
        done_chest_imaging = any(node in self.asked_node_ids for node in [79, 81])

        # 基础过滤（噪声绝对屏蔽）
        PNEUMONIA_NOISE = [51, 52, 54, 55, 56, 57, 58, 72, 73, 74, 76, 82, 84, 85, 88, 89, 90, 91, 92]

        for i in range(self.num_nodes):
            if i == DIAG_NODE or i in PNEUMONIA_NOISE:
                continue
            if (48 <= i <= 92) and i in self.asked_node_ids:
                continue

            unasked = self.retriever.get_unasked_candidates(i, self.asked_entities)
            if len(unasked) > 0:
                mask[i] = 1.0

        # ==========================================
        # 4. 强制临床路径约束（坚持你的铁律）
        # ==========================================
        # 只有在化验和胸部影像【双双齐备】的情况下，才允许诊断！
        mask[DIAG_NODE] = 1.0 if (done_labs and done_chest_imaging) else 0.0

        # ==========================================
        # 5. 阶段性解禁与【强制推进】
        # ==========================================
        if current_turn is not None:
            if current_turn < 1:
                # 第 0 轮：纯问诊，禁止一切检查和诊断
                mask[48:93] = 0.0
                mask[DIAG_NODE] = 0.0

            # 👇 核心新增：关门打狗机制！
            # 如果已经聊到第 4 轮了，核心证据还没凑齐，说明 Agent 在墨迹。
            # 直接把 0-47（问诊节点）全部屏蔽，逼它去点化验和影像！
            elif current_turn >= 4 and not (done_labs and done_chest_imaging):
                mask[0:48] = 0.0

                # ==========================================
        # 6. 健壮性兜底与智能引导
        # ==========================================
        if mask.sum() == 0 or (current_turn is not None and current_turn >= 2):
            available_labs = [i for i in LAB_NODES if i not in self.asked_node_ids and i not in PNEUMONIA_NOISE]
            available_imaging = [i for i in IMAGING_NODES if i not in self.asked_node_ids and i not in PNEUMONIA_NOISE]

            if not done_labs and available_labs:
                forced_labs = [i for i in [59, 71] if i not in self.asked_node_ids]
                target_labs = forced_labs if forced_labs else available_labs[:3]
                if target_labs:
                    mask[target_labs] = 1.0

            if not done_chest_imaging and available_imaging:
                forced_imaging = [i for i in [79, 81] if i not in self.asked_node_ids]
                target_imaging = forced_imaging if forced_imaging else available_imaging[:3]
                if target_imaging:
                    mask[target_imaging] = 1.0

        return mask

    def _get_mask_pure_inquiry(self, current_turn=None, max_turns=None, gender=None, age=None):
        """
        🔥 模式1：纯问诊模式（HLA 仅限 0-47）
        """
        mask = torch.zeros(self.num_nodes).to(self.device)
        DIAG_NODE = self.DIAGNOSIS_NODE_ID

        # 1. 仅允许 0-47 问诊节点
        for i in range(48):
            if gender == 'male' and i in [44, 45]:
                continue
            if age is not None and age < 14 and i in [41, 42, 45]:
                continue
            if i in self.asked_node_ids:
                continue

            unasked = self.retriever.get_unasked_candidates(i, self.asked_entities)
            if len(unasked) > 0:
                mask[i] = 1.0

        # 2. 诊断节点控制
        if current_turn is not None:
            if current_turn >= 7:          # 可根据实验调整阈值
                mask[DIAG_NODE] = 1.0
            if mask[:48].sum() == 0:       # 问诊节点耗尽 → 强制诊断
                mask[DIAG_NODE] = 1.0

        if current_turn is not None and max_turns is not None and current_turn >= max_turns:
            mask[DIAG_NODE] = 1.0

        # 3. 熔断保护（防连续重复）
        if len(self.hla_history) >= 1:
            last = self.hla_history[-1]
            if last != DIAG_NODE:
                mask[last] = 0.0

        if mask.sum() == 0:
            mask[DIAG_NODE] = 1.0

        return mask

    def _get_mask_inquiry_plus_physical(self, current_turn=None, max_turns=None, gender=None, age=None):
        """
        🔥 模式2：症状 + 体查模式（HLA 允许 0-58）
        新增约束：前三轮（0-2轮）仅允许问诊（0-47），第3轮起放开体查（48-58）
        """
        mask = torch.zeros(self.num_nodes).to(self.device)
        DIAG_NODE = self.DIAGNOSIS_NODE_ID

        # ==========================================
        # 1. 节点允许范围（带轮次控制）
        # ==========================================
        max_allowed = 47  # 默认只允许问诊

        if current_turn is not None and current_turn >= 3:
            max_allowed = 58  # 第3轮起允许体查

        for i in range(max_allowed + 1):  # 0 到 max_allowed
            # 性别/年龄常识过滤
            if gender == 'male' and i in [44, 45]:
                continue
            if age is not None and age < 14 and i in [41, 42, 45]:
                continue

            # 重复动作屏蔽
            if i in self.asked_node_ids:
                continue

            # 检查该 HLA 下是否还有未问过的 LLA 实体
            unasked = self.retriever.get_unasked_candidates(i, self.asked_entities)
            if len(unasked) > 0:
                mask[i] = 1.0

        # ==========================================
        # 2. 诊断节点控制（体查模式下可适当提前）
        # ==========================================
        if current_turn is not None:
            # 至少问够一定轮数（体查模式建议第4轮起可诊断）
            if current_turn >= 4:
                mask[DIAG_NODE] = 1.0

            # 如果当前允许范围内的节点都已问完，强制允许诊断
            if mask[:max_allowed + 1].sum() == 0:
                mask[DIAG_NODE] = 1.0

        # 强制结束条件
        if current_turn is not None and max_turns is not None and current_turn >= max_turns:
            mask[DIAG_NODE] = 1.0

        # ==========================================
        # 3. 熔断保护（防止连续重复同一动作）
        # ==========================================
        if len(self.hla_history) >= 1:
            last = self.hla_history[-1]
            if last != DIAG_NODE:
                mask[last] = 0.0

        # 兜底：没有任何动作可选时强制诊断
        if mask.sum() == 0:
            mask[DIAG_NODE] = 1.0

        return mask
    # def select_hla_action(self, state_tensor, current_turn=None, max_turns=None, gender=None, age=None):
    #     # 增加背景参数
    #     valid_mask = self.get_hla_mask(current_turn, max_turns, gender, age)
    #
    #     if valid_mask[self.DIAGNOSIS_NODE_ID] == 1.0 and valid_mask.sum() == 1:
    #         action = self.DIAGNOSIS_NODE_ID
    #     elif random.random() <= self.epsilon:
    #         indices = torch.nonzero(valid_mask).flatten().tolist()
    #         action = random.choice(indices) if indices else self.DIAGNOSIS_NODE_ID
    #     else:
    #         with torch.no_grad():
    #             q = self.hla_net(state_tensor).squeeze(0)
    #             masked_q = q.masked_fill(valid_mask == 0, -1e9)
    #             action = torch.argmax(masked_q).item()
    #
    #     self.hla_history.append(action)
    #     return action
    def select_hla_action(self, state_tensor, current_turn=None, max_turns=None, gender=None, age=None):
        """HLA动作选择：适配不同限制模式"""
        valid_mask = self.get_hla_mask(current_turn, max_turns, gender, age)

        # 调试信息
        num_available = valid_mask[:59].sum().item()  # 统计可见范围
        diag_available = valid_mask[self.DIAGNOSIS_NODE_ID].item()
        mode_name = self.hla_mode if hasattr(self, 'hla_mode') else 'full'

        # print(f"🔍 [HLA {mode_name}] 当前轮次: {current_turn} | "
        #       f"可用节点: {num_available} | 诊断可用: {diag_available}")

        if valid_mask[self.DIAGNOSIS_NODE_ID] == 1.0 and valid_mask.sum() == 1:
            action = self.DIAGNOSIS_NODE_ID
        elif random.random() <= self.epsilon:
            indices = torch.nonzero(valid_mask).flatten().tolist()
            action = random.choice(indices) if indices else self.DIAGNOSIS_NODE_ID
        else:
            with torch.no_grad():
                q = self.hla_net(state_tensor).squeeze(0)
                masked_q = q.masked_fill(valid_mask == 0, -1e9)
                # 🔥 【核心修改：Q-value Bonus 引导机制】
                # 条件 1：诊断灯亮了 (说明核心证据如血常规、胸片已集齐)
                # 条件 2：当前轮次达到了你认为“可以收手”的阶段 (比如第 4 轮及以后)
                if valid_mask[self.DIAGNOSIS_NODE_ID] == 1.0 and current_turn is not None and current_turn >= 6:
                    # 给诊断节点的 Q 值人为加上一个“诱惑分”
                    # 因为你的满分是 100，这里给 15.0 ~ 20.0 能够有效扭转偏好
                    diag_bonus = 20.0
                    masked_q[self.DIAGNOSIS_NODE_ID] += diag_bonus

                    # 可选：如果你想看它有没有触发引导，可以取消下面这行注释
                    # print(f"🚀 [Q-Bonus 触发] 轮次: {current_turn} | 诊断Node原Q值: {q[self.DIAGNOSIS_NODE_ID].item() commentary:.2f} -> 加成后: {masked_q[self.DIAGNOSIS_NODE_ID].item():.2f}")

                action = torch.argmax(masked_q).item()

                self.hla_history.append(action)
        return action
    #DLBCL探索激励
    # def select_hla_action(self, state_tensor, current_turn=None, max_turns=None, gender=None, age=None):
    #     """HLA动作选择：epsilon-greedy + Mask + 检查探索激励（优化版）"""
    #     valid_mask = self.get_hla_mask(current_turn, max_turns, gender, age)
    #
    #     # 强制诊断场景（保持不变）
    #     if valid_mask[self.DIAGNOSIS_NODE_ID] == 1.0 and valid_mask.sum() == 1:
    #         action = self.DIAGNOSIS_NODE_ID
    #     elif random.random() <= self.epsilon:
    #         # 随机探索阶段：增加检查节点的采样权重
    #         indices = torch.nonzero(valid_mask).flatten().tolist()
    #         if indices:
    #             # 🔥 探索激励：检查节点更高概率
    #             weights = []
    #             for idx in indices:
    #                 if self.is_exam_node(idx):  # 确保这个方法已定义
    #                     weights.append(1.8 if current_turn is None or current_turn < 6 else 1.5)
    #                 else:
    #                     weights.append(1.0)
    #             action = random.choices(indices, weights=weights, k=1)[0]
    #         else:
    #             action = self.DIAGNOSIS_NODE_ID
    #     else:
    #         # 贪心选择阶段
    #         with torch.no_grad():
    #             q = self.hla_net(state_tensor).squeeze(0)
    #             masked_q = q.masked_fill(valid_mask == 0, -1e9)
    #
    #             # 🔥 Q-value bonus（动态调整）
    #             bonus = 3.0 if (current_turn is None or current_turn < 8) else 2.0
    #             for i in range(len(masked_q)):
    #                 if (valid_mask[i] == 1.0 and
    #                         self.is_exam_node(i) and
    #                         i not in self.asked_node_ids):
    #                     masked_q[i] = masked_q[i] + bonus
    #
    #             action = torch.argmax(masked_q).item()
    #
    #     self.hla_history.append(action)
    #
    #     # **调试日志（带保护，防止logger未定义）**
    #     if current_turn is not None and current_turn % 3 == 0:
    #         hla_name = self.get_node_name(action) if hasattr(self, 'get_node_name') else f"ID_{action}"
    #         exam_count = sum(1 for a in self.hla_history if self.is_exam_node(a))
    #
    #         # try:
    #         #     logger.info(f"[HLA Turn {current_turn}] 选中: {hla_name} (id={action}) | "
    #         #                 f"epsilon={self.epsilon:.3f} | 本episode已开检查: {exam_count}次")
    #         # except NameError:
    #         #     print(f"[HLA Turn {current_turn}] 选中: {hla_name} (id={action}) | "
    #         #           f"epsilon={self.epsilon:.3f} | 本episode已开检查: {exam_count}次")
    #         # except Exception as e:
    #         #     print(f"[HLA Debug Warning] {str(e)}")
    #
    #     return action

    def select_lla_action(self, state_tensor, hla_node_id, temperature=2.0):
        """
        修正说明：
        1. 增加 Embedding 归一化：防止输入模长过大导致 Logits 爆炸。
        2. 增加 Logits Clamping：限制得分范围在 [-10, 10]，强制保留探索空间。
        3. 返回 Entropy：用于在 update_model 中真正实现最大化熵，解决 LogProb 锁死。
        """
        target_dtype = next(self.lla_critic.parameters()).dtype
        state_tensor = state_tensor.to(device=self.device, dtype=target_dtype)
        valid_candidates = self.retriever.get_unasked_candidates(hla_node_id, self.asked_entities)
        state_value = self.lla_critic(state_tensor)

        if not valid_candidates or len(valid_candidates) == 0:
            return None, None, state_value, torch.tensor(0.0).to(self.device)

        # 1. 归一化输入特征 (防止由于 LLM Embedding 数值过大导致的策略坍缩)
        state_norm = F.normalize(state_tensor, p=2, dim=-1)
        cand_tensor = self.backbone.get_candidate_embeddings(valid_candidates)
        cand_norm = F.normalize(cand_tensor, p=2, dim=-1)

        state_exp = state_norm.repeat(len(valid_candidates), 1)

        # 2. 计算得分并进行裁剪 (Clamping)
        scores = self.lla_actor(state_exp, cand_norm).squeeze(1)
        # 强制将原始得分限制在合理范围，防止 Softmax 后概率直接变成 1.0
        scores = torch.clamp(scores, min=-10.0, max=10.0)

        # 3. 应用 Temperature Softmax
        probs = F.softmax(scores / temperature, dim=0)

        # 4. 采样与概率计算
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()

        log_prob = dist.log_prob(idx)
        entropy = dist.entropy()  # 🔥 获取分布熵用于后续训练

        entity = valid_candidates[idx.item()]

        # 记录已问
        self.asked_entities.add(entity)
        if hasattr(self, 'mutex_pairs') and entity in self.mutex_pairs:
            self.asked_entities.add(self.mutex_pairs[entity])

        return entity, log_prob, state_value, entropy

    def predict_diagnosis(self, state_tensor, temperature=1.0):
        """
        预测诊断结果，增加温度缩放(Temperature Scaling)以提升置信度。
        :param temperature: 温度系数，越小(如0.5)分布越尖锐(自信)，越大(如2.0)分布越平滑。
        """
        # 🔥 核心修改：动态对齐数据类型 (Dtype Alignment)
        # 获取诊断头当前的数据类型 (通常是 float32)
        target_dtype = next(self.diag_head.parameters()).dtype

        # 将输入的 state_tensor 强制转换为一致的类型
        state_tensor = state_tensor.to(dtype=target_dtype)

        # 然后再进行前向传播
        logits = self.diag_head(state_tensor)

        # 🔥 核心修改：缩放 logits
        # 注意：这里我们只缩放 logits，不在这里做 softmax
        # 因为外部调用者(如 predict_stats)通常会自己做 softmax
        # 或者使用 CrossEntropyLoss (它内部自带 softmax)
        # 但为了让外部拿到的概率更高，我们在这里直接返回缩放后的 logits 是最简单的办法
        scaled_logits = logits / temperature

        return scaled_logits

    def get_node_name(self, node_id):
        if self.node_df is None: return f"Node_{node_id}"
        try:
            target_id = str(int(node_id))
            row = self.node_df[self.node_df['node_id'].astype(str) == target_id]
            if not row.empty: return str(row.iloc[0]['node_name']).strip()
            return f"Node_{target_id}(Unknown)"
        except:
            return f"Node_{node_id}(Error)"

    def is_exam_node(self, node_id):
        n = self.get_node_name(node_id)
        return ("辅查" in n or "检查" in n or "体查" in n)

    def generate_natural_question(self, hla_id, lla_entity=None, mode="train"):
        raw_name = self.get_node_name(hla_id)
        node_name = raw_name.split('-')[-1].split('(')[0].replace(")", "")
        if lla_entity:
            return f"请问您有{lla_entity}的情况吗？"
        if self.is_exam_node(hla_id):
            return f"(医生正在检查{node_name}...)"
        return f"请问{node_name}方面有什么不舒服吗？"

    def mark_whole_node_as_asked(self, hla_id):
        """
        [V1.2 核心修复]
        双重封禁逻辑：同时封锁 HLA 节点和下属所有 LLA 实体
        """
        # 1. 封锁 HLA 节点本身（解决之前 get_hla_mask 报错的问题）
        # 这样下一次 select_hla_action 时，Mask 会自动过滤掉这个 ID
        if not hasattr(self, 'asked_node_ids'):
            self.asked_node_ids = set()
        self.asked_node_ids.add(hla_id)

        # 2. 封锁该 HLA 节点下属的所有候选 LLA 实体（保持原有逻辑）
        # 这样模型就不会在 LLA 层级重复询问具体细节
        all_cands = self.retriever.get_all_candidates(hla_id)
        if all_cands:
            self.asked_entities.update(all_cands)