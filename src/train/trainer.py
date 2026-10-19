# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import random
import logging
import argparse
import shutil
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# 如果只想过滤这个特定消息：
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*")
# ==========================================================
# 🟢 1. 环境配置
# ==========================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

try:
    from src.doctor.rl import DoctorAgent
    from src.patient.simulator import PatientSimulator
except ImportError:
    sys.path.append(os.path.join(project_root, "src"))
    from doctor.rl import DoctorAgent
    from patient.simulator import PatientSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ==========================================================
# 🟢 2. 工具类
# ==========================================================
def split_and_clean_response(text):
    if not text:
        return []
    text = re.sub(r'^.*?[：:]', '', text)  # ++++ 兼容中英文冒号
    segments = re.split(r'[;；。\n]', text)
    clean_segs = [s.strip() for s in segments if len(s.strip()) > 1]
    return clean_segs

import json
import os

class ResponseCache:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"📦 [Cache] 已加载现有缓存，条目数: {len(self.cache)}")
            except Exception as e:
                print(f"⚠️ [Cache] 加载失败: {e}")

    def _get_key(self, case_id, action_name):
        # 唯一键：病例ID + 动作名称
        return f"{case_id}_{action_name}"

    def get(self, case_id, action_name):
        key = self._get_key(case_id, action_name)
        return self.cache.get(key)

    def set(self, case_id, action_name, response):
        key = self._get_key(case_id, action_name)
        self.cache[key] = response

    def save(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        print("🧹 Replay buffer cleared.")


class CheckpointManager:
    def __init__(self, save_dir, name):
        self.name = name
        self.save_dir = os.path.join(save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        self.latest_ckpt_path = os.path.join(save_dir, f"checkpoint_latest_{name}.pth")
        self.best_model_path = os.path.join(save_dir, f"doctor_best_{name}.pth")
        self.emergency_path = os.path.join(save_dir, f"checkpoint_EMERGENCY_{name}.pth")

    def save_checkpoint(self, state_dict, is_best=False, is_emergency=False):
        # 👇 【核心新增】一进函数，不管三七二十一，先确保 name 对应的子目录绝对存在
        os.makedirs(self.save_dir, exist_ok=True)

        if is_emergency:
            save_path = self.emergency_path
            logger.warning(f"🚨 正在执行紧急保存 -> {save_path}")
        else:
            save_path = self.latest_ckpt_path

        tmp_path = save_path + ".tmp"
        try:
            torch.save(state_dict, tmp_path)
        except Exception as e:
            logger.error(f"❌ 保存临时文件失败: {e}")
            return

        if os.path.exists(save_path):
            try:
                backup_path = save_path + ".bak"
                shutil.copy2(save_path, backup_path)
            except:
                pass

        try:
            os.replace(tmp_path, save_path)
            if not is_emergency:
                logger.info(f"💾 Checkpoint saved: Epoch {state_dict.get('epoch', '?')}")
        except OSError as e:
            logger.error(f"❌ Checkpoint rename failed: {e}")

        if not is_emergency:
            current_epoch = state_dict.get('epoch', -1) + 1
            if current_epoch > 0 and current_epoch % 5 == 0:
                history_filename = f"checkpoint_ep{current_epoch}_{self.name}.pth"
                history_path = os.path.join(self.save_dir, history_filename)
                try:
                    if os.path.exists(save_path):
                        # 🔄 这里就可以安全地复制了，因为上面已经把目录建好了
                        shutil.copy2(save_path, history_path)
                        logger.info(f"🗂️ Periodic backup (Ep {current_epoch}) saved inside subfolder: {history_filename}")
                except Exception as e:
                    logger.error(f"❌ Failed to save periodic backup: {e}")

        if is_best:
            model_weights = state_dict['model_state_dict']
            torch.save(model_weights, self.best_model_path)
            logger.info(f"🏆 Best model updated: {self.best_model_path}")

    def load_latest(self, device):
        path_to_load = None
        if os.path.exists(self.latest_ckpt_path):
            path_to_load = self.latest_ckpt_path
        elif os.path.exists(self.emergency_path):
            path_to_load = self.emergency_path

        if path_to_load:
            logger.info(f"🔄 Resuming from checkpoint: {path_to_load}")
            try:
                checkpoint = torch.load(path_to_load, map_location=device)
                return checkpoint
            except Exception as e:
                logger.error(f"❌ Checkpoint file corrupted: {e}")
                return None
        return None

    # ==========================================================
    # 🟢 3. HybridTrainer
    # ==========================================================
class HybridTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 日志 ---
        self.log_dir = os.path.join(project_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        self.log_file = os.path.join(self.log_dir, f"train_log_{getattr(args, 'name', 'exp')}_{timestamp}.jsonl")

        self.ckpt_manager = CheckpointManager(args.save_dir, args.name)
        run_name = f"{getattr(args, 'name', 'exp')}_phase2" if getattr(args, 'phase2', False) else getattr(args,
                                                                                                           'name',
                                                                                                           'exp')
        tb_dir = os.path.join(project_root, "runs", f"tb_{timestamp}_{run_name}")
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_dir)
        print("👨‍⚕️ 初始化 AI 医生 (Struct Memory Mode)...")

        # 初始化缓存文件路径（建议放在数据目录下）
        cache_dir = os.path.join(project_root, "src", "data","processed", "DLBCL","cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.sim_cache = ResponseCache(os.path.join(cache_dir, f"sim_cache_{args.data}.json"))

        # ++++ 统一根据 --data 参数设置路径
        data_processed_dir = os.path.join(project_root, "src", "data", "processed", args.data)
        label_path = os.path.join(data_processed_dir, "disease_labels.json")  # ++++
        csv_path = os.path.join(data_processed_dir, "wen.csv")  # ++++

        # 🔥 修改点: 动态获取模型简称，与训练诊断头时的命名逻辑保持完全一致
        # 假设 train.py 的 args 中也有 model_path 参数
        # 🔥 使用 --model_path 决定诊断头权重（推荐方案）
        model_name_short = "Qwen2.5"
        if hasattr(args, 'model_path') and args.model_path:
            # 从完整路径中提取文件夹名
            path_basename = os.path.basename(str(args.model_path).rstrip('/'))
            if 'glm' in path_basename.lower() or 'chatglm' in path_basename.lower():
                model_name_short = "GLM-4-9B-Chat"
            elif 'qwen' in path_basename.lower():
                model_name_short = "Qwen2.5"
        else:
            # 兜底方案：如果没传 model_path，则根据 --name 参数判断
            name_str = getattr(args, 'name', '').lower()
            if 'glm' in name_str or 'chatglm' in name_str:
                model_name_short = "GLM-4-9B-Chat"
            elif 'qwen' in name_str:
                model_name_short = "Qwen2.5"

        # diag_weight_path = os.path.join(
        #     project_root,
        #     "src",
        #     "model",
        #     "checkpoints",
        #     f"diag_head_{model_name_short}_{args.data}_best.pth"
        # )
                # 🟢 如果是第二阶段，初始化底座诊断头时依然使用 IIYI（中文）的离线诊断头作为基准
        data_folder = "IIYI" if args.phase2 else args.data
        diag_weight_path = os.path.join(
            project_root,
            "src",
            "model",
            "checkpoints",
            f"diag_head_{model_name_short}_{data_folder}_best.pth"
        )
        print(f"✅ 将加载预训练诊断头权重: {diag_weight_path} （model: {model_name_short}）")


        # 初始化 Doctor
        self.doctor = DoctorAgent(args=args,label_path=label_path, csv_path=csv_path, model_weight_path=diag_weight_path)
        self.doctor.train()
        if getattr(args, 'phase2', False):
            print("🔧 [Config] Phase 2: 执行策略精修阶段...")
            # 策略层（HLA/LLA）：负责“什么时候开检查”，需要比第一阶段更活跃一些
            backbone_lr = 1e-4  # 原 1e-1 改为 1e-4
            # 诊断层（Diag Head）：负责“最后定什么病”，需要稳中求进
            diag_head_lr = 1e-5  # 原 2e-6 改为 3e-5)
        else:
            print("🚀 [Config] Phase 1: 执行基础训练阶段...")
            backbone_lr = 5e-5
            diag_head_lr = 1e-4  #1e-6

        self.optimizer = optim.Adam([
            {'params': self.doctor.hla_net.parameters(), 'lr': backbone_lr},
            {'params': self.doctor.lla_actor.parameters(), 'lr': backbone_lr},
            {'params': self.doctor.lla_critic.parameters(), 'lr': backbone_lr},
            {'params': self.doctor.diag_head.parameters(), 'lr': diag_head_lr}
        ])

        buffer_size = 1500 if getattr(args, 'phase2', False) else 5000
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.gamma = 0.99
        self.batch_size = 32
        self.entropy_coef = 0.05

        # Patient Model
        # self.patient_model_path = "/data/pretrained_models/Qwen2.5-7B-Instruct"
        # print(f"📥 [Trainer] Loading Patient Model from: {self.patient_model_path} ...")
        #
        # try:
        #     self.patient_tokenizer = AutoTokenizer.from_pretrained(self.patient_model_path, trust_remote_code=True)
        #     self.patient_model = AutoModelForCausalLM.from_pretrained(
        #         self.patient_model_path,
        #         trust_remote_code=True,
        #         torch_dtype=torch.float16,
        #         attn_implementation="sdpa",
        #         device_map=None
        #     ).to(self.device)
        #     self.patient_model.eval()
        #     print("✅ Patient Model Loaded Successfully!")
        # except Exception as e:
        #     print(f"❌ Error loading patient model: {e}")
        #     sys.exit(1)
        self.patient_model_path = "/data/pretrained_models/Qwen2.5-7B-Instruct"
        print(f"📥 [Trainer] Loading Patient Model from: {self.patient_model_path} (4bit 量化)...")

        try:
            # ==================== 4bit 量化配置 ====================
            from transformers import BitsAndBytesConfig  # 如果文件顶部未导入，请添加

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # 与你测试效果匹配，推荐使用 bfloat16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            # ============================================================

            self.patient_tokenizer = AutoTokenizer.from_pretrained(
                self.patient_model_path,
                trust_remote_code=True
            )

            self.patient_model = AutoModelForCausalLM.from_pretrained(
                self.patient_model_path,
                quantization_config=quant_config,  # 关键：加入量化配置
                trust_remote_code=True,
                device_map="auto",  # 推荐改为 auto，量化模型更友好
                # torch_dtype=torch.float16,           # 量化后可注释掉
                # attn_implementation="sdpa",          # 量化模式下可先去掉测试
            ).eval()

            # device_map="auto" 已自动处理设备分配，无需额外 .to(self.device)
            print("✅ Patient Model 4bit 量化加载成功！")

        except Exception as e:
            print(f"❌ Error loading patient model: {e}")
            sys.exit(1)
        # ==========================================================
        # 🔥 修改点：核心 Reward 与平衡参数 (按照要求修改)
        # ==========================================================
        self.top1_reward = getattr(args, "top1_reward", 100.0) # 正确诊断大奖
        self.top3_reward = getattr(args, "top3_reward", 30.0)  # Top-3 命中奖励
        self.wrong_diag_penalty = getattr(args, "wrong_diag_penalty", 30.0)  # 错误诊断重罚
        self.step_cost = -2.0  # 统一时间成本
        self.prob_beta = getattr(args, "prob_beta", 5.0)  # 置信度增益权重 (驱动核心)
        self.exam_forbidden_penalty = -50.0  # 前3轮禁开检查重罚
        self.diag_conf_thres = float(getattr(args, "diag_conf_thres", 0.55))

    def _compute_confidence_reward(self, prev_prob, true_idx):
        """计算置信度提升奖励"""
        curr_true, _, _, _ = self._predict_stats(true_idx)
        if curr_true is None:
            return 0.0
        d = self._logit(curr_true) - self._logit(prev_prob)
        return max(d, 0.0) * self.prob_beta

    def _logit(self, p):
        """辅助函数：防止log(0)"""
        p = max(min(p, 1 - 1e-6), 1e-6)
        return np.log(p / (1 - p))

    def _get_node_name(self, node_id):
        return self.doctor.get_node_name(node_id)

    def save_log(self, record):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"Log save failed: {e}")

    def _analyze_response_sentiment(self, response_text, mode="inquiry"):
        text = response_text.strip()
        if not text: return 0.0, False

        # ---------------------------------------------------------
        # 检查模式 (体格检查 / 辅助检查)
        # ---------------------------------------------------------
        if mode == "exam":
            # 1. 首先判定“无数据” (无奖励)
            if any(kw in text for kw in ["未见该项", "未见记录", "无记录", "不详", "未做", "未查"]):
                return -1.0, False

            # 2. 【核心修改】先判定“正常/阴性” (判定为阴性 False)
            # 优先级最高，防止被后面的“异常”关键词误拦截
            normal_kws = ["正常", "未见异常", "未见明显异常", "阴性", "整齐", "清晰", "未触及", "无压痛"]
            if any(kw in text for kw in normal_kws):
                return 1.0, False

            # 3. 再判定“异常/阳性” (判定为阳性 True)
            abnormal_kws = ["异常", "偏低", "偏高", "啰音", "障碍", "降低", "升高", "增高", "增大",
                            "阴影", "结节", "渗出", "感染", "炎", "粗糙", "满布"]
            if any(kw in text for kw in abnormal_kws):
                return 1.0, True

            # 4. 默认兜底：如果既没写正常也没写异常（通常是有具体数值，如血压140/90）
            # 这种通常视为有意义的发现
            return 1.0, True

        # ---------------------------------------------------------
        # 问诊模式 (主诉 / 现病史)
        # ---------------------------------------------------------
        else:
            # 1. 判定否定含义（阴性）
            # 增加对“无”、“否认”等词的覆盖
            neg_kws = ["没有", "未", "否认", "无", "正常", "没有过"]
            if any(kw in text for kw in neg_kws):
                return 1.0, False

            # 2. 判定肯定含义（阳性）
            pos_kws = ["有", "是", "对", "痛", "咳", "热", "喘", "觉得", "感觉"]
            if any(kw in text for kw in pos_kws):
                return 1.0, True

            # 3. 兜底逻辑：如果包含主诉中的英文关键词或描述，也视为阳性
            return 1.0, True

    @staticmethod
    def _logit(p: float, eps: float = 1e-6) -> float:
        p = float(np.clip(p, eps, 1 - eps))
        return float(np.log(p / (1 - p)))

    def _predict_stats(self, true_idx: int):
        curr_struct_embed = self.doctor.get_structured_state_for_diagnosis()
        with torch.no_grad():
            logits = self.doctor.predict_diagnosis(curr_struct_embed)
            probs = F.softmax(logits, dim=1)
            topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
            top3 = topk.indices[0].tolist()
            pred_idx = int(top3[0])
            top1_prob = float(topk.values[0, 0].item())
            true_prob = float(probs[0, true_idx].item())
            return true_prob, top1_prob, top3, pred_idx

    # ============================================================
    # 🟢 Train Episode (重构 Reward 逻辑)
    # ============================================================
    def train_episode(self, case_data):
        simulator = PatientSimulator(
            patient_data=case_data,
            model=self.patient_model,
            tokenizer=self.patient_tokenizer,
            device=self.device
        )
        self.doctor.reset()
        asked_entities_ids = set()
        hla_history = []

        raw_info = str(case_data['raw_medical_record'].get('一般资料', '')) + \
                   str(case_data['raw_medical_record'].get('主诉', '')) + \
                   str(case_data['raw_medical_record'].get('现病史', ''))
        lang = 'en' if len(re.findall(r'[a-zA-Z]', raw_info)) > 0.3 * len(raw_info) else 'zh'
        if lang == 'zh':
                patient_gender = 'male' if '男' in raw_info else 'female'
                age_match = re.search(r'(\d+)岁', raw_info)
                patient_age = int(age_match.group(1)) if age_match else 30
                opening_q = "您好，请问哪里不舒服？"

        else:  # 英文病历
                raw_lower = raw_info.lower()
                patient_gender = 'female' if any(w in raw_lower for w in ['female', 'woman', 'girl', ' f ', 'sex: f']) else 'male'
                age_match = re.search(r'(?:age\s*)?(\d+)\s*(?:years?\s*old|y\.?o\.?|y/o)', raw_lower)
                if not age_match:
                    age_match = re.search(r'(\d+)[- ]?(?:year|yr|y\.o)', raw_lower)
                patient_age = int(age_match.group(1)) if age_match else 30
                # 英文开场白，引导 Simulator 用中文回答
                opening_q = ("你好，我是你的主治医生，请问今天哪里不舒服？"
                             "(Please strictly answer in Chinese based on your medical record / 请务必根据病历用纯中文简短回答主诉)")
        opening_a = simulator.step(opening_q)  # 现在返回 str，不再是 dict
        history = f"医生：{opening_q}\n患者：{opening_a}"

        extracted = self.doctor.parse_observation(opening_a, turn_idx=0)
        self.doctor.update_struct_memory(extracted, turn_idx=0)
######调试
        # logger.info(f"\n{'=' * 60}")
        # logger.info(f"🔍 [Turn 0 开场] 自然语言: {opening_a[:200]}...")
        # logger.info(f"📋 Parsed extracted: {extracted}")
        # logger.info(f"🧠 Current struct_memory: {self.doctor.struct_memory}")
        # logger.info(f"{'=' * 60}\n")
        curr_state = self.doctor.get_state_embedding(history).detach()

        dialog_history = [{"turn": 0, "role": "Doctor", "content": opening_q},
                          {"turn": 0, "role": "Patient", "content": opening_a}]
        total_reward = 0.0
        done = False
        lla_trajectory = []
        supervised_loss = 0.0

        is_success_strict = False
        is_in_top3 = False
        display_turn = 1
        true_idx = int(case_data['diagnosis_idx'])
        init_true_prob, init_top1_prob, init_top3, init_pred_idx = self._predict_stats(true_idx)
        prev_true_prob = init_true_prob if init_true_prob is not None else 0.0

        for t in range(self.args.max_turns + 2):
            curr_state_np = curr_state.cpu().detach().numpy()[0]

            # 动态时间惩罚：随步数微增，促使模型不拖拉
            # time_penalty = -1.0 - (t * 0.1)

            if t >= self.args.max_turns:
                dialog_history.append(
                    {"turn": display_turn, "role": "Debug_Log", "content": "【系统强制】已达最大轮数"})
                hla_action = self.doctor.DIAGNOSIS_NODE_ID
                hla_name = "⚠️ 强制诊断"
            else:
                # 调用带物理 Mask 感知的 HLA 选择
                hla_action = self.doctor.select_hla_action(curr_state, current_turn=t,
                                                           max_turns=self.args.max_turns,
                                                           gender=patient_gender, age=patient_age)
                hla_name = self._get_node_name(hla_action)
                dialog_history.append(
                    {"turn": display_turn, "role": "Debug_Log", "content": f"【模型决策】HLA选中: {hla_name}"})

            hla_history.append(hla_action)
            step_reward = 0.0
            turn_done = False
            # =================================================================================
            # Branch A: Diagnosis (诊断分支)
            # =================================================================================
            if hla_action == self.doctor.DIAGNOSIS_NODE_ID:

                curr_true_prob, curr_top1_prob, curr_top3, pred_idx = self._predict_stats(true_idx)
                # &&&
                # print("\n" + "=" * 50)
                # print("🧠 [Diagnosis Head 准备诊断] 它此时看到的结构化记忆是：")
                # # 如果你的 get_structured_state_for_diagnosis 返回的是文本，直接打印：
                # print(self.doctor.get_structured_state_for_diagnosis())
                # print("=" * 50 + "\n")
                # [🔥 核心修改]：删除动态门槛，统一使用固定的 self.diag_conf_thres
                # print("\n" + "🩺 " * 10 + "诊断时刻内存校验" + " 🩺" * 10)
                # # 1. 打印原始字典，看提取的实体对不对
                # print(f"【结构化字典 (Dictionary)】: {self.doctor.struct_memory}")
                #
                # # 2. 打印序列化后的文本，看喂给 Backbone 的字符串格式是否正确
                # # 注意：_get_memory_text_representation 是私有方法，但在 Python 中可以直接调用用于调试
                # memory_text = self.doctor._get_memory_text_representation()
                # print(f"【输入文本 (String)】: {memory_text}")

                # 3. 打印当前诊断头预测出的 Top1 和 真实标签，看模型是否被误导
                # 这里你代码里已经有 _predict_stats 了，我们可以直接用它的结果
                curr_true_prob, curr_top1_prob, curr_top3, pred_idx = self._predict_stats(true_idx)
                pred_name = self.doctor.idx_to_disease_name.get(pred_idx, "Unknown")
                true_name = case_data.get('diagnosis', "Unknown")
                # print(f"【预测结果】: {pred_name} (概率: {curr_top1_prob:.4f})")
                # print(f"【正确诊断】: {true_name}")
                # print("—" * 60 + "\n")
                is_forced = (t >= self.args.max_turns)

                if not is_forced and curr_top1_prob < self.diag_conf_thres:
                    step_reward = -10.0
                    dialog_history.append({
                        "turn": display_turn, "role": "Debug_Log",
                        "content": f"🛑 诊断被门槛拦截: {curr_top1_prob:.2f} < {self.diag_conf_thres:.2f}"
                    })
                    turn_done = False
                else:
                    pred_name = self.doctor.idx_to_disease_name.get(pred_idx, f"Unknown_{pred_idx}")
                    true_name_local = case_data.get('diagnosis', f"Unknown_{true_idx}")

                    if true_idx == pred_idx:
                        # [🔥 核心修改]：删除 early_turn_bonus 和 active_diag_bonus，仅保留最纯粹的终极奖励
                        step_reward = self.top1_reward
                        is_success_strict = True
                        is_in_top3 = True
                    elif true_idx in curr_top3:
                        step_reward = self.top3_reward
                        is_in_top3 = True
                    else:
                        step_reward = -self.wrong_diag_penalty
                    #强制快速诊断
                    if is_forced:
                        step_reward -= 50.0
                        dialog_history.append(
                            {"turn": display_turn, "role": "Debug_Log", "content": "⚠️ 触发超时重罚 (-50)"})
                    msg = "⚠️ [强制诊断]" if is_forced else "✅ [主动诊断]"
                    dialog_history.append({"turn": display_turn, "role": "Doctor",
                                           "content": f"{msg} #诊断结果# {pred_name} (Prob: {curr_top1_prob:.2f})\n#真实诊断# {true_name_local}"})

                    clean_state = self.doctor.get_structured_state_for_diagnosis()
                    logits = self.doctor.predict_diagnosis(clean_state)
                    target = torch.tensor([true_idx]).long().to(self.device)
                    supervised_loss = F.cross_entropy(logits, target)
                    turn_done = True
            # =================================================================================
            # Branch B: Exam (辅助检查/体查分支)
            # =================================================================================
            elif self.doctor.is_exam_node(hla_action):
                if hla_action in asked_entities_ids:
                    step_reward = -15.0
                else:
                    self.doctor.mark_whole_node_as_asked(hla_action)
                    asked_entities_ids.add(hla_action)
                    short_name = str(hla_name).split('-')[-1].split('(')[0]
                    action_cost = -1.5

                    # 🔥 中英双语映射：用动作名匹配英文病历中的实体关键字
                    raw_record_str = str(case_data['raw_medical_record']).lower()
                    short_name_lower = short_name.lower()
                    keywords = [short_name_lower]

                    if any(x in short_name_lower for x in ["x-ray", "radiograph", "x光", "胸片", "xray"]):
                        keywords.extend(["chest film", "dr", "pa view", "lateral view", "cxr", "x-ray"])
                    elif any(x in short_name_lower for x in ["ct", "断层"]):
                        keywords.extend(["computed tomography", "chest ct", "hrct", "ct scan"])
                    elif any(x in short_name_lower for x in ["cbc", "blood routine", "血常规", "白细胞"]):
                        keywords.extend(["wbc", "white blood cell", "hemoglobin", "platelet", "neutrophil", "rbc"])
                    elif any(x in short_name_lower for x in ["ecg", "ekg", "心电图"]):
                        keywords.extend(["st segment", "t wave", "rhythm", "sinus", "electrocardiogram"])
                    elif any(x in short_name_lower for x in ["chemistry", "metabolic", "生化", "肝功", "肾功"]):
                        # 👇 关键新增：加入 ldh, lactate dehydrogenase 等淋巴瘤核心指标
                        keywords.extend(["alt", "ast", "creatinine", "bun", "urea", "transaminase", "cmp", "bmp", "ldh",
                                         "lactate dehydrogenase", "liver function"])
                    # 👇 [新增修改 1]：增加高级影像、内镜与病理/穿刺/活检的英文病历关键字映射
                    elif any(x in short_name_lower for x in ["pet", "pet-ct", "petct"]):
                        keywords.extend(["pet", "fdg", "suv", "metabolism", "hypermetabolic", "uptake"])
                    elif any(x in short_name_lower for x in
                             ["endoscopy", "gastroscopy", "colonoscopy", "内镜", "胃镜", "肠镜"]):
                        keywords.extend(["endoscopy", "scope", "mucosa", "ulcer", "mass", "biopsy taken"])
                    elif any(x in short_name_lower for x in
                             ["pathology", "biopsy", "活检", "病理", "骨髓", "穿刺", "细胞学"]):
                        keywords.extend(
                            ["pathology", "biopsy", "marrow", "histology", "immunohistochemistry", "ihc", "cd20",
                             "malignant", "carcinoma", "lymphoma", "cytology", "flow cytometry"])

                    evidence_snippet = ""
                    for kw in keywords:
                        if kw in raw_record_str:
                            evidence_snippet = f" (提示: 包含'{kw}'相关记录)"
                            break
                    question = self.doctor.generate_natural_question(hla_action, lla_entity=None, mode="train")
                    # 将提示隐式传递给大模型
                    # &&& 修改点 2：强制大模型将检查结果输出为中文 &&&
                    sim_input = question + (evidence_snippet if evidence_snippet else "")
                    sim_input += "\n(Please strictly reply in Chinese / 请务必用纯中文回答此检查结果)"

                    # --- 💡 新增 Cache 逻辑 ---
                    case_id = case_data.get('case_id', 'unknown')
                    hla_name = self.doctor.get_node_name(hla_action)
                    action_key = f"exam_{hla_name}"

                    cached_res = self.sim_cache.get(case_id, action_key)
                    if cached_res:
                        nl_response = cached_res
                    else:
                        nl_response = simulator.step(sim_input)  # 现在返回 str
                        self.sim_cache.set(case_id, action_key, nl_response)
                    # --- Cache 逻辑结束 ---

                    history += f"\n医生：{question}\n患者：{nl_response}"
                    dialog_history.append({"turn": display_turn, "role": "Doctor", "content": question})
                    dialog_history.append({"turn": display_turn, "role": "Patient", "content": nl_response})

                    extracted = self.doctor.parse_observation(
                        nl_response,
                        turn_idx=display_turn,
                        prev_memory=self.doctor.struct_memory,
                        doctor_question=question
                    )
                    self.doctor.update_struct_memory(extracted, turn_idx=display_turn)
######调试
                    # logger.info(f"\n{'-' * 50}")
                    # logger.info(f"🔍 [Turn {display_turn} 检查] 问题: {question}")
                    # logger.info(f"📝 患者回答: {nl_response[:300]}...")
                    # logger.info(f"📋 Parsed extracted: {extracted}")
                    # logger.info(f"🧠 Updated struct_memory: {self.doctor.struct_memory}")
                    # logger.info(f"{'-' * 50}\n")
                    curr_true, _, _, _ = self._predict_stats(true_idx)
                    conf_rew = self._compute_confidence_reward(prev_true_prob, true_idx) * 2.0   # 检查证据放大系数 2.0

                    has_pos = False
                    if extracted and isinstance(extracted, dict):
                        pos_findings = extracted.get("positive", []) or extracted.get("abnormal", []) or extracted.get(
                            "阳性", [])
                        has_pos = len(pos_findings) > 0
                    elif any(k in nl_response for k in ["异常", "阳性", "升高", "病变", "确诊"]):
                        has_pos = True

                    finding_reward = 0.0
                    clean_n = short_name.replace(")", "").strip().lower()
                    if has_pos:
                        if any(x in clean_n for x in ["pathology", "biopsy", "活检", "病理", "骨髓", "穿刺", "细胞学"]):
                            finding_reward = 18.0
                        elif any(x in clean_n for x in ["肝功", "生化", "ldh"]):
                            finding_reward = 9.0
                        else:
                            finding_reward = 5.0
                    exam_bonus = 4.0 if has_pos else 1.5
                    step_reward = action_cost + finding_reward + conf_rew + exam_bonus
                    prev_true_prob = max(prev_true_prob, curr_true)  # 更新 prev
                    display_turn += 1
            # =================================================================================
            # Branch C: Inquiry (问诊分支)
            # =================================================================================
            else:
                entity, log_prob, value, entropy = self.doctor.select_lla_action(curr_state, hla_action)
                if entity:
                    if entity in asked_entities_ids:
                        step_reward = -10.0
                    else:
                        asked_entities_ids.add(entity)
                        ent_name = str(entity)
                        dialog_history.append({"turn": display_turn, "role": "Debug_Log",
                                               "content": f"【LLA决策】实体: {entity}(LogProb: {log_prob:.4f})"})

                        action_cost = -1.0
                        question = self.doctor.generate_natural_question(hla_action, entity,
                                                                         mode="train")

                        # &&& 修改点 4：强制大模型将问诊症状输出为中文 &&&
                        sim_input = question + "\n(Please strictly reply in Chinese / 请务必根据你的英文病历用纯中文回答)"

                        # step_result = simulator.step(sim_input)
                        # --- 💡 新增 Cache 逻辑 ---
                        case_id = case_data.get('case_id', 'unknown')
                        action_key = f"inquiry_{entity}"

                        cached_res = self.sim_cache.get(case_id, action_key)
                        if cached_res:
                            nl_response = cached_res
                        else:
                            nl_response = simulator.step(sim_input)  # 现在直接返回 str
                            self.sim_cache.set(case_id, action_key, nl_response)# 4. 存入
                        # --- Cache 逻辑结束 ---

                        # 从返回的字典中安全提取对应的值
                        history += f"\n医生：{question}\n患者：{nl_response}"
                        dialog_history.append(
                            {"turn": display_turn, "role": "Doctor", "content": question})
                        dialog_history.append(
                            {"turn": display_turn, "role": "Patient", "content": nl_response})

                        extracted = self.doctor.parse_observation(
                            nl_response,
                            turn_idx=display_turn,
                            prev_memory=self.doctor.struct_memory,
                            doctor_question=question
                        )
                        self.doctor.update_struct_memory(extracted, turn_idx=display_turn)
                        # logger.info(f"\n{'-' * 50}")
                        # logger.info(f"🔍 [Turn {display_turn} 问诊] 问题: {question}")
                        # logger.info(f"📝 患者回答: {nl_response[:300]}...")
                        # logger.info(f"📋 Parsed extracted: {extracted}")
                        # logger.info(f"🧠 Updated struct_memory: {self.doctor.struct_memory}")
                        # logger.info(f"{'-' * 50}\n")
                        curr_true, _, _, _ = self._predict_stats(true_idx)
                        conf_rew = self._compute_confidence_reward(prev_true_prob, true_idx) * 0.8
                        has_pos = False
                        if extracted and isinstance(extracted, dict):
                            pos_findings = extracted.get("positive", []) or extracted.get("主诉", []) or extracted.get(
                                "阳性", [])
                            has_pos = len(pos_findings) > 0
                        elif any(k in nl_response for k in ["有", "疼痛", "不适", "症状", "出现"]):
                            has_pos = True

                        finding_reward = 2.0 if has_pos else 0.0
                        step_reward = action_cost + finding_reward + conf_rew
                        prev_true_prob = max(prev_true_prob, curr_true)

                        lla_trajectory.append(
                            {"log_prob": log_prob, "value": value, "reward": step_reward,
                             "entropy": entropy})
                        display_turn += 1
                else:
                    step_reward = -2.0

                    # 👇========== 【核心新增：全局步数惩罚】 =========👇
                    # 只要当前动作不是“下诊断”，就强行扣除一点费用
                    # 这就相当于病人的“时间成本”，防止模型赖着不走
            if hla_action != self.doctor.DIAGNOSIS_NODE_ID:
                dynamic_penalty = -1 - (t * 0.2)   # 极小的固定惩罚（或者启用你之前注释的动态惩罚）
                step_reward += dynamic_penalty
            clipped_reward = max(min(float(step_reward), 50.0), -50.0)
            total_reward += float(step_reward)
            next_state = curr_state if turn_done else self.doctor.get_state_embedding(f"{history}\n{hla_name}").detach()

            next_state = curr_state if turn_done else self.doctor.get_state_embedding(f"{history}\n{hla_name}").detach()
            mask = self.doctor.get_hla_mask(t + 1, self.args.max_turns, patient_gender, patient_age)
            self.buffer.push(
                curr_state_np,
                hla_action,
                clipped_reward,  # 🔥 这里用截断后的奖励
                next_state.cpu().numpy()[0],
                1 if turn_done else 0,
                mask.cpu().numpy().astype(bool)
            )
            curr_state = next_state
            if turn_done: break
        if random.random() < 0.1:
            self.sim_cache.save()
        loss_val = self.update_model(lla_trajectory, supervised_loss)
        self.save_log({
            "patient_id": case_data.get("case_id", 0),
            "result": "Success" if is_success_strict else "Fail",
            "is_top3": is_in_top3, "turns": t, "dialog_history": dialog_history, "reward": total_reward
        })
        return total_reward, loss_val, (1 if is_success_strict else 0), (1 if is_in_top3 else 0), t, 0, 0, 0

    def update_model(self, lla_traj, diag_loss):
        losses = []  # 推荐做法：用列表收集不同分支的 loss，最后统一 sum，避免计算图断裂

        # ============================================================
        # Part 1: DQN Loss —— HLA 网络更新
        # ============================================================
        if self.buffer.size() > self.batch_size:
            s, a, r, ns, d, nmask = self.buffer.sample(self.batch_size)

            # ✅ 优化 1：使用 torch.as_tensor 避免不必要的内存拷贝
            s = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
            a = torch.as_tensor(np.array(a), dtype=torch.long, device=self.device)
            r = torch.as_tensor(np.array(r), dtype=torch.float32, device=self.device)
            ns = torch.as_tensor(np.array(ns), dtype=torch.float32, device=self.device)
            d = torch.as_tensor(np.array(d), dtype=torch.float32, device=self.device)
            nmask = torch.as_tensor(np.array(nmask), dtype=torch.bool, device=self.device)

            r_scaled = r * 0.01

            q_values = self.doctor.hla_net(s)
            q_current = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next_all = self.doctor.target_hla_net(ns)
                q_next_all = q_next_all.masked_fill(~nmask, -1e9)
                q_next = q_next_all.max(dim=1).values
                q_next = torch.where(nmask.any(dim=1), q_next, torch.zeros_like(q_next))
                target = r_scaled + self.gamma * q_next * (1 - d)

            dqn_loss_val = F.smooth_l1_loss(q_current, target, reduction='mean')
            losses.append(dqn_loss_val * 0.4)

        # ============================================================
        # Part 2: Actor-Critic Loss —— LLA 网络更新
        # ============================================================
        if len(lla_traj) > 0:
            rewards = [item['reward'] * 0.01 for item in lla_traj]
            values = torch.cat([item['value'] for item in lla_traj])

            # Monte Carlo Returns
            returns = []
            R = 0.0
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # ✅ 优化 2：使用 view(-1) 代替 squeeze()，防止 0-d 张量报错
            advantages = returns - values.detach().view(-1)

            # ✅ 优化 3：使用 numel() 代替 len()，提高鲁棒性
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            advantages = torch.clamp(advantages, -3.0, 3.0)

            actor_loss = torch.tensor(0.0, device=self.device)
            critic_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)

            for i, item in enumerate(lla_traj):
                critic_loss += F.smooth_l1_loss(
                    item['value'], returns[i].view(1, 1), reduction='mean'
                )
                actor_loss += -item['log_prob'] * advantages[i]
                if 'entropy' in item and item['entropy'] is not None:
                    entropy_loss += -item['entropy']

            n = len(lla_traj)
            ac_loss_val = (
                    actor_loss / n
                    + critic_loss / n * 0.5
                    + entropy_loss / n * self.entropy_coef
            )
            losses.append(ac_loss_val * 0.25)

        # ============================================================
        # Part 3: Supervised Diagnosis Loss —— 诊断头更新
        # ============================================================
        if isinstance(diag_loss, torch.Tensor):
            current_eps = getattr(self.doctor, 'epsilon', 0.5)

            # 例如：eps 为 0.9 时，权重约 1.0；eps 为 0.1 时，权重约 7.4
            dynamic_weight = 1.0 + (1.0 - current_eps) * 8.0

            losses.append(diag_loss * dynamic_weight)

        # ============================================================
        # Backward & Optimize
        # ============================================================
        # ✅ 优化 4：只要收集到了 loss 就统一 sum 并执行反向传播
        if losses:
            total_loss = sum(losses)

            if total_loss.requires_grad:
                self.optimizer.zero_grad()
                total_loss.backward()

                grad_params = [p for p in self.doctor.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)

                self.optimizer.step()

                # 软更新 target_hla_net
                tau = 0.005
                for param, target_param in zip(
                        self.doctor.hla_net.parameters(),
                        self.doctor.target_hla_net.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                return total_loss.item()

        return 0.0

    def close(self):
        try:
            if self.tb_writer is not None:
                self.tb_writer.flush();
                self.tb_writer.close()
        except Exception:
            pass
if __name__ == "__main__":
    logging.getLogger("transformers").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="IIYI")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--max_turns", type=int, default=10)
    default_save_path = os.path.join(project_root, "src", "train", "models")
    parser.add_argument("--save_dir", type=str, default=default_save_path)
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--eps_start", type=float, default=0.95)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--name", type=str, default="v1")
    parser.add_argument("--phase2", action="store_true", help="是否开启第二阶段微调")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Backbone LLM 模型路径，用于确定 diag_head 权重文件名")
    # 1.置信度门槛
    parser.add_argument("--diag_conf_thres", type=float, default=0.40)
    # 2.诊断奖励
    parser.add_argument("--top1_reward", type=float, default=120.0)
    parser.add_argument("--top3_reward", type=float, default=30.0)
    parser.add_argument("--wrong_diag_penalty", type=float, default=60.0)
    # 3.信息增益奖励
    parser.add_argument("--prob_beta", type=float, default=5.0)
    parser.add_argument("--entropy_beta", type=float, default=2.0)
    parser.add_argument("--entropy_min_gain", type=float, default=0.01)
    parser.add_argument("--entropy_bad_beta", type=float, default=2.0)
    # 4.采样训练设置
    parser.add_argument("--warmup_sample", type=int, default=100, help="初期采样病例数")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="前多少个 epoch 进行采样")
    # 5.恢复训练
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='指定要恢复的 checkpoint 完整路径，为 None 则使用 latest')
    args = parser.parse_args()

    # Phase 2 Config Override
    if args.phase2:
        print("🚀 [System] 正在激活 Phase 2 配置模式...")
        best_path = os.path.join(args.save_dir, "doctor_best_qwen_IIYI_v1.0.pth")
        if not args.load_path and os.path.exists(best_path):
            args.load_path = best_path
            print(f"🔄 [Phase 2] 自动锁定上一阶段最佳权重: {best_path}")

        args.eps_start = 0.4
        args.eps_end = 0.05
        args.diag_conf_thres = 0.55
        args.entropy_beta = 1.0
        args.top1_reward = 150.0
        args.wrong_diag_penalty = 80.0
        args.early_diag_bonus_coef = 1.0
        args.evidence_value_coef = 10.0
        args.prob_beta = 5.0

    data_path = os.path.join(project_root, "src", "data", "processed",args.data, f"{args.data}_cases_train.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(project_root, "src", "data", "processed", "data.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        all_cases = json.load(f)
    train_cases = [c for c in all_cases if 'diagnosis_idx' in c]

    trainer = HybridTrainer(args)

    if args.phase2:
        trainer.diag_conf_thres = args.diag_conf_thres
        trainer.top1_reward = args.top1_reward
        trainer.wrong_diag_penalty = args.wrong_diag_penalty

    # ==========================================
    # 🔥 自动恢复逻辑 (支持指定 checkpoint)
    # ==========================================
    start_epoch = 0
    best_top1_acc = 0.0

    # --- 新增：Phase 2 启动时强制清空旧经验 ---
    if args.phase2 and start_epoch == 0:
        print("🧨 [Phase 2] 检测到新阶段开始，正在强制清空旧 Buffer 数据...")
        trainer.buffer.clear()

    # ==================== 优先使用指定 checkpoint ====================
    resume_path = None
    if hasattr(args, 'resume_checkpoint') and args.resume_checkpoint:
        resume_path = args.resume_checkpoint
        print(f"🔄 [Resume] 使用指定 checkpoint: {resume_path}")
    else:
        # 回退到原来的 latest 逻辑
        latest_ckpt = trainer.ckpt_manager.load_latest(trainer.device)
        if latest_ckpt:
            print("⚡ [Resume] 发现 latest 断点，正在恢复训练状态...")
            resume_path = "latest"  # 标记使用 latest

    if resume_path:
        try:
            if resume_path == "latest":
                checkpoint = latest_ckpt
            else:
                # 加载指定文件
                checkpoint = torch.load(resume_path, map_location=trainer.device)

            # 恢复模型权重
            trainer.doctor.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # 恢复优化器状态（如果存在）
            if 'optimizer_state_dict' in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 恢复 epoch 和最佳指标
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_top1_acc = checkpoint.get('best_top1_acc', 0.0)

            print(f"   ✅ 成功恢复至 Epoch {start_epoch - 1} | Best Top-1: {best_top1_acc:.2%}")
            print(f"   📍 当前 epsilon 将从调度对应位置继续")

        except Exception as e:
            print(f"❌ Resume 失败: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    else:
        # 2. 如果没有 checkpoint，但有 load_path (手动加载权重)
        if args.load_path and os.path.exists(args.load_path):
            print(f"🔄 [Load] 加载指定权重: {args.load_path}")
            state_dict = torch.load(args.load_path, map_location=trainer.device)
            if 'model_state_dict' in state_dict:
                trainer.doctor.load_state_dict(state_dict['model_state_dict'], strict=False)
            else:
                trainer.doctor.load_state_dict(state_dict, strict=False)
        elif args.phase2:
            print("❌ [Error] Phase 2 模式下未找到 load_path！")
            sys.exit(1)
        else:
            print("🚀 从头开始训练（未找到任何 checkpoint）")
    # =====================================================================

    # ==========================================
    # 🟢 训练循环 (带异常捕获)
    # ==========================================
    # os.makedirs(args.save_dir, exist_ok=True)

    # # ==================== 探索能力强化版 epsilon 调度 ====================
    # # 针对你的目标（提升探索选择质量）：前重探索、后期温和衰减
    # total_epochs = args.epochs
    # stage1_end = int(total_epochs * 0.4)  # 前40%：强探索
    # stage2_end = int(total_epochs * 0.8)  # 中间40%：维持较高探索（核心强化区间）
    #
    # print(f"🚀 探索强化调度 | 前{stage1_end}轮强探索 | 中间维持较高水平 | 后{total_epochs - stage2_end}轮温和衰减")
    # print(f"   eps_start={args.eps_start}, eps_end={args.eps_end}")
    #
    # try:
    #     use_full_data = False
    #     ep_top1 = 0.0
    #     for epoch in range(start_epoch, args.epochs):
    #         # ==================== Epsilon 三阶段计算 ====================
    #         if epoch < stage1_end:
    #             # 前40%：从 eps_start 缓慢下降，仍保持较高探索
    #             progress = epoch / max(stage1_end, 1)
    #             current_eps = args.eps_start - (args.eps_start - 0.50) * progress
    #         elif epoch < stage2_end:
    #             # 中间40%：稳定在较高探索水平（重点强化探索能力）
    #             current_eps = 0.45
    #         else:
    #             # 后20%：温和线性下降到 eps_end
    #             progress = (epoch - stage2_end) / max(total_epochs - stage2_end, 1)
    #             current_eps = 0.45 - (0.45 - args.eps_end) * progress
    #
    #         current_eps = max(current_eps, args.eps_end)  # 保底
    #         trainer.doctor.epsilon = current_eps
    # 针对你的目标（提升探索选择质量）：30%快速下降 -> 30%平缓下降 -> 40%低探索微调
    # total_epochs = args.epochs
    # stage1_end = int(total_epochs * 0.3)  # 前30%：快速结束强探索期
    # stage2_end = int(total_epochs * 0.6)  # 中间30%：平缓衰减，过渡到利用期
    #
    # print(f"🚀 探索强化调度 | 前{stage1_end}轮快速下降 | 中间{stage2_end - stage1_end}轮平缓衰减 | 后{total_epochs - stage2_end}轮低探索微调")
    # print(f"   eps_start={args.eps_start}, eps_end={args.eps_end}")
    #
    # try:
        # use_full_data = False
        # ep_top1 = 0.0
        # for epoch in range(start_epoch, args.epochs):
        #     # ==================== Epsilon 三阶段计算(优化版) ====================
        #     if epoch < stage1_end:
        #         # 阶段一（前30%）：从 eps_start 快速下降到 0.30
        #         progress = epoch / max(stage1_end, 1)
        #         current_eps = args.eps_start - (args.eps_start - 0.30) * progress
        #     elif epoch < stage2_end:
        #         # 阶段二（中间30%）：从 0.30 缓慢下降到 0.10
        #         progress = (epoch - stage1_end) / max(stage2_end - stage1_end, 1)
        #         current_eps = 0.30 - (0.30 - 0.10) * progress
        #     else:
        #         # 阶段三（后40%）：从 0.10 线性下降到 eps_end (专注策略微调)
        #         progress = (epoch - stage2_end) / max(total_epochs - stage2_end, 1)
        #         current_eps = 0.10 - (0.10 - args.eps_end) * progress
        #
        #     current_eps = max(current_eps, args.eps_end)  # 保底
        #     trainer.doctor.epsilon = current_eps
        # 把原先计算 0.3 的地方改成与你代码一致：
    # 1. 预计算每个阶段的长度 (用于记录日志)
    stage1_end = int(args.epochs * 0.20)  # 快速下降期截止
    stage2_end = int(args.epochs * 0.50)  # 缓慢下降期截止

    stage1_len = stage1_end
    stage2_len = stage2_end - stage1_end
    stage3_len = args.epochs - stage2_end

    logger.info(f"🚀 探索强化调度 | 前{stage1_len}轮快速下降 | 中间{stage2_len}轮平缓衰减 | 后{stage3_len}轮低探索微调")

    try:
        use_full_data = False
        ep_top1 = 0.0

        for epoch in range(start_epoch, args.epochs):
            # ==========================================================
            # 💡 核心优化：使用 epoch 作为绝对进度，保证断点恢复后 Epsilon 不抖动
            # ==========================================================
            if epoch < stage1_end:
                # 阶段一：从 eps_start 快速下降到 0.20
                # 进度条：从 0 到 1
                progress = epoch / max(stage1_end, 1)
                current_eps = args.eps_start - (args.eps_start - 0.20) * progress

            elif epoch < stage2_end:
                # 阶段二：从 0.20 缓慢下降到 0.05
                # 进度条：从 0 到 1 (epoch 在 stage1_end 到 stage2_end 之间)
                progress = (epoch - stage1_end) / max(stage2_end - stage1_end, 1)
                current_eps = 0.20 - (0.20 - 0.05) * progress

            else:
                # 阶段三：从 0.05 线性下降到 eps_end
                # 进度条：从 0 到 1 (epoch 在 stage2_end 到 总epochs 之间)
                progress = (epoch - stage2_end) / max(args.epochs - stage2_end, 1)
                current_eps = 0.05 - (0.05 - args.eps_end) * progress

            # 保底：确保任何情况下 epsilon 不会低于用户设置的终点值
            current_eps = max(current_eps, args.eps_end)
            trainer.doctor.epsilon = current_eps

            # =====================================================================
            # # ==================== 基于“剩余训练轮数”的高探索调度 ====================
            # remaining_epochs = args.epochs - start_epoch  # 关键：从当前恢复点算起
            # if remaining_epochs <= 0:
            #     remaining_epochs = 20  # 安全兜底
            #
            # high_exp_epochs = int(remaining_epochs * 0.70)  # 前70% 保持高探索
            #
            # # 当前 epoch 相对于恢复起点的相对位置
            # relative_epoch = epoch - (start_epoch - 1)  # 从 1 开始计数
            #
            # if relative_epoch < high_exp_epochs:
            #     # 前70%：极高探索，缓慢下降
            #     progress = relative_epoch / max(high_exp_epochs, 1)
            #     current_eps = args.eps_start - (args.eps_start - 0.75) * progress  # 目标高值可调至0.75
            # else:
            #     # 后30%：温和衰减，保留一定探索
            #     progress = (relative_epoch - high_exp_epochs) / max(remaining_epochs - high_exp_epochs, 1)
            #     current_eps = 0.75 - (0.75 - args.eps_end) * progress
            #
            # current_eps = max(current_eps, args.eps_end)
            # trainer.doctor.epsilon = current_eps
            # =====================================================================
            # ============================================================

            random.shuffle(train_cases)
            # ==========================================
            # 🔥 核心修改：动态采样逻辑
            # ==========================================
            if not use_full_data:
                # 触发切换条件：达到 warmup 轮数 OR 准确率达标
                if epoch >= args.warmup_epochs or (epoch > start_epoch and ep_top1 > 0.75):
                    use_full_data = True
                    print(f"🎊 [Signal] 模式切换：性能达标或达到预设轮数，进入全量训练！")

            if not use_full_data:
                current_train_set = random.sample(train_cases, args.warmup_sample)
                print(f"⚡ [Warmup] Epoch {epoch + 1}: 随机抽样 {len(current_train_set)} 例")
            else:
                current_train_set = train_cases
                random.shuffle(current_train_set)  # 🔥 修正 2: 只在全量模式 shuffle 即可
                print(f"🌟 [Full] Epoch {epoch + 1}: 全量训练 {len(current_train_set)} 例")

            total_reward = 0.0;
            total_loss = 0.0
            correct_top1 = 0;
            correct_top3 = 0;
            total_turns = 0

            # pbar = tqdm(current_train_set, desc=f"Ep {epoch + 1}/{args.epochs} [eps={current_eps:.2f}]", ncols=120)
            # +++ 【修改 1：精简前缀并锁死安全宽度】 +++
            pbar = tqdm(
                current_train_set,
                desc=f"Ep {epoch + 1}/{args.epochs}",  # 去掉了长长的 [eps=...]，让左边更短
                ncols=100,  # 🔥 核心：锁死总宽度为 100 个字符，绝对不会折行！
                leave=False,  # 🔥 当前 epoch 跑完后隐藏进度条，只留下一行干净的总结
                file=sys.stdout
            )
            # +++++++++++++++++++++++++++++++++++++++++++
            for i, case in enumerate(pbar):
                r, l, t1, t3, turns, _, _, _ = trainer.train_episode(case)
                total_reward += r;
                total_loss += l
                correct_top1 += t1;
                correct_top3 += t3;
                total_turns += turns

                pbar.set_postfix({
                    'R': f"{r:.1f}",
                    'L': f"{l:.2f}",
                    'Acc': f"{correct_top1 / (i + 1):.2f}",
                    'Turn': f"{total_turns / (i + 1):.1f}"  # 新增
                })
            # Metrics
            num_actual = len(current_train_set)
            ep_top1 = correct_top1 / num_actual
            ep_top3 = correct_top3 / num_actual
            ep_rew = total_reward / num_actual
            ep_avg_turns = total_turns / num_actual  # 确保计算了这个变量

            print(
                f"\n📊 Ep {epoch + 1}: Rew: {ep_rew:.2f} | Top-1: {ep_top1:.2%} | Top-3: {ep_top3:.2%} | Avg Turns: {ep_avg_turns:.1f}")

            # TensorBoard
            trainer.tb_writer.add_scalar("train/avg_reward", ep_rew, epoch)
            trainer.tb_writer.add_scalar("train/top1_acc", ep_top1, epoch)
            trainer.tb_writer.add_scalar("train/top3_acc", ep_top3, epoch)
            trainer.tb_writer.add_scalar("train/epsilon", current_eps, epoch)
            trainer.tb_writer.add_scalar("train/avg_turns", ep_avg_turns, epoch)

            # 🔥 Save Checkpoint (Periodic)
            is_best = False
            if ep_top1 > best_top1_acc:
                best_top1_acc = ep_top1
                is_best = True

            # 构建完整状态字典
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': trainer.doctor.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_top1_acc': best_top1_acc,
                'epsilon': current_eps,
                'args': vars(args)
            }

            # 每 epoch 自动保存断点 + 最佳模型
            trainer.ckpt_manager.save_checkpoint(checkpoint_state, is_best=is_best)

    except Exception as e:
        print(f"\n🚨 发生严重错误: {e}")
        import traceback

        traceback.print_exc()

        # 🔥 异常时的紧急保存
        print("💾 正在执行紧急保存 (Emergency Save)...")
        emergency_state = {
            'epoch': epoch if 'epoch' in locals() else start_epoch,
            'model_state_dict': trainer.doctor.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_top1_acc': best_top1_acc,
            'note': 'Emergency save due to crash'
        }
        trainer.ckpt_manager.save_checkpoint(emergency_state, is_emergency=True)
        print("✅ 紧急保存完成，请检查 checkpoint_EMERGENCY 文件。")
        sys.exit(1)

    finally:
        trainer.close()
        print("👋 程序已退出")# -*- coding: utf-8 -*-
