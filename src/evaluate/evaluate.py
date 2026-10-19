# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import argparse
import logging
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ResponseCache:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"📦 [Cache] 已加载缓存: {len(self.cache)}")
            except Exception as e:
                print(f"⚠️ [Cache] 加载失败: {e}")

    def _get_key(self, case_id, action_name):
        return f"{case_id}_{action_name}"

    def get(self, case_id, action_name):
        return self.cache.get(self._get_key(case_id, action_name))

    def set(self, case_id, action_name, response):
        self.cache[self._get_key(case_id, action_name)] = response

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_processed_dir = os.path.join(project_root, "src", "data", "processed", args.data)
        label_path = os.path.join(data_processed_dir, "disease_labels.json")
        csv_path = os.path.join(data_processed_dir, "wen.csv")

        print("👨‍⚕️ 初始化 DoctorAgent...")
        self.doctor = DoctorAgent(
            args=args,
            label_path=label_path,
            csv_path=csv_path,
            model_weight_path=None
        )
        self.doctor.hla_mode = args.hla_mode
        print(f"🔧 [Eval] 当前评估强制设定 DoctorAgent HLA 模式为: {self.doctor.hla_mode}")
        self.doctor.to(self.device)
        self.doctor.eval()
        self.doctor.epsilon = 0.0

        self.load_doctor_weights(args.load_path)

        cache_dir = os.path.join(data_processed_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.sim_cache = ResponseCache(os.path.join(cache_dir, f"eval_sim_cache_{args.data}.json"))

        self.patient_model_path = args.patient_model_path
        print(f"📥 加载 Patient Model: {self.patient_model_path} (4bit)")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.patient_tokenizer = AutoTokenizer.from_pretrained(
            self.patient_model_path,
            trust_remote_code=True
        )

        self.patient_model = AutoModelForCausalLM.from_pretrained(
            self.patient_model_path,
            quantization_config=quant_config,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

        print("✅ Eval 初始化完成")

    # def load_doctor_weights(self, load_path):
    #     if not load_path or not os.path.exists(load_path):
    #         raise FileNotFoundError(f"未找到模型权重: {load_path}")
    #
    #     print(f"🔄 加载 Doctor 权重: {load_path}")
    #     state = torch.load(load_path, map_location=self.device)
    #
    #     if isinstance(state, dict) and "model_state_dict" in state:
    #         state_dict = state["model_state_dict"]
    #     else:
    #         state_dict = state
    #
    #     missing, unexpected = self.doctor.load_state_dict(state_dict, strict=False)
    #     print(f"✅ 权重加载完成 | missing={len(missing)} | unexpected={len(unexpected)}")
    def load_doctor_weights(self, load_path):
        if not load_path or not os.path.exists(load_path):
            raise FileNotFoundError(f"未找到模型权重: {load_path}")

        print(f"🔄 加载 Doctor 权重: {load_path}")
        state = torch.load(load_path, map_location=self.device)

        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state

        # 🎯 精准匹配你打印出来的 [512, 256, 7] 架构
        if "diag_head.net.0.weight" in state_dict:
            checkpoint_hidden_dim = state_dict["diag_head.net.0.weight"].shape[0]  # 512
            current_hidden_dim = self.doctor.diag_head.net[0].weight.shape[0]  # 1024

            if checkpoint_hidden_dim != current_hidden_dim:
                print(f"⚠️ [Fix] 检测到维度不匹配! 权重维度: {checkpoint_hidden_dim} vs 模型维度: {current_hidden_dim}")
                print(f"🛠️ 正在精准重构 self.doctor.diag_head.net 为纯三层全连接架构...")

                in_features = self.doctor.diag_head.net[0].in_features  # 3584
                out_features = self.doctor.diag_head.net[-1].out_features  # 7

                # 🔄 严格按照 0, 3, 6 的索引顺序和层类型进行重构
                self.doctor.diag_head.net = torch.nn.Sequential(
                    torch.nn.Linear(in_features, checkpoint_hidden_dim),  # net.0 (3584 -> 512)
                    torch.nn.ReLU(),  # net.1
                    torch.nn.Dropout(0.3),  # net.2 (如果有的话，不影响索引)
                    torch.nn.Linear(checkpoint_hidden_dim, 256),  # net.3 (512 -> 256)
                    torch.nn.ReLU(),  # net.4
                    torch.nn.Dropout(0.3),  # net.5
                    torch.nn.Linear(256, out_features)  # net.6 (256 -> 7)
                ).to(self.device)

        # 严格检查加载情况
        missing, unexpected = self.doctor.load_state_dict(state_dict, strict=False)
        print(f"✅ 权重加载完成 | missing={len(missing)} | unexpected={len(unexpected)}")
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"⚠️ 仍有未对齐参数！missing keys: {missing} | unexpected keys: {unexpected}")


    def _get_node_name(self, node_id):
        return self.doctor.get_node_name(node_id)

    def _predict_stats(self, true_idx):
        clean_state = self.doctor.get_structured_state_for_diagnosis()
        with torch.no_grad():
            logits = self.doctor.predict_diagnosis(clean_state)
            probs = F.softmax(logits, dim=1)
            topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
            top3 = topk.indices[0].tolist()
            pred_idx = int(top3[0])
            top1_prob = float(topk.values[0, 0].item())
            true_prob = float(probs[0, true_idx].item())
        return true_prob, top1_prob, top3, pred_idx

    def _memory_json(self):
        mem = getattr(self.doctor, "struct_memory", {})
        try:
            return json.dumps(mem, ensure_ascii=False)
        except Exception:
            return str(mem)

    def _detect_patient_info(self, case_data):
        raw_info = (
            str(case_data["raw_medical_record"].get("一般资料", "")) +
            str(case_data["raw_medical_record"].get("主诉", "")) +
            str(case_data["raw_medical_record"].get("现病史", ""))
        )

        lang = "en" if len(re.findall(r"[a-zA-Z]", raw_info)) > 0.3 * max(len(raw_info), 1) else "zh"

        if lang == "zh":
            gender = "male" if "男" in raw_info else "female"
            age_match = re.search(r"(\d+)岁", raw_info)
            age = int(age_match.group(1)) if age_match else 30
            opening_q = "您好，请问哪里不舒服？"
        else:
            raw_lower = raw_info.lower()
            gender = "female" if any(
                w in raw_lower for w in ["female", "woman", "girl", " f ", "sex: f"]
            ) else "male"
            age_match = re.search(r"(?:age\s*)?(\d+)\s*(?:years?\s*old|y\.?o\.?|y/o)", raw_lower)
            if not age_match:
                age_match = re.search(r"(\d+)[- ]?(?:year|yr|y\.o)", raw_lower)
            age = int(age_match.group(1)) if age_match else 30
            opening_q = (
                "你好，我是你的主治医生，请问今天哪里不舒服？"
                "(Please strictly answer in Chinese based on your medical record / 请务必根据病历用纯中文简短回答主诉)"
            )

        return gender, age, opening_q

    def _sim_step_cached(self, simulator, case_id, action_key, sim_input):
        cached = self.sim_cache.get(case_id, action_key)
        if cached:
            return cached

        res = simulator.step(sim_input)
        self.sim_cache.set(case_id, action_key, res)
        return res

    def evaluate_case(self, case_data):
        simulator = PatientSimulator(
            patient_data=case_data,
            model=self.patient_model,
            tokenizer=self.patient_tokenizer,
            device=self.device
        )

        self.doctor.reset()
        self.doctor.eval()
        self.doctor.epsilon = 0.0

        asked_entities_ids = set()
        dialog_history = []

        patient_gender, patient_age, opening_q = self._detect_patient_info(case_data)
        opening_a = simulator.step(opening_q)

        history = f"医生：{opening_q}\n患者：{opening_a}"

        dialog_history.append({"turn": 0, "role": "Doctor", "content": opening_q})
        dialog_history.append({"turn": 0, "role": "Patient", "content": opening_a})

        extracted = self.doctor.parse_observation(opening_a, turn_idx=0)
        self.doctor.update_struct_memory(extracted, turn_idx=0)

        dialog_history.append({
            "turn": 0,
            "role": "Debug_Log",
            "content": f"🧠 【实时记忆更新】: {self._memory_json()}"
        })

        curr_state = self.doctor.get_state_embedding(history).detach()

        true_idx = int(case_data["diagnosis_idx"])
        is_success = False
        is_top3 = False
        display_turn = 1

        for t in range(self.args.max_turns + 1):
            if t >= self.args.max_turns:
                dialog_history.append({
                    "turn": display_turn - 1,
                    "role": "Debug_Log",
                    "content": "【系统强制】已达最大轮数"
                })
                hla_action = self.doctor.DIAGNOSIS_NODE_ID
                hla_name = "⚠️ 强制诊断"
                is_forced = True
            else:
                with torch.no_grad():
                    hla_action = self.doctor.select_hla_action(
                        curr_state,
                        current_turn=t,
                        max_turns=self.args.max_turns,
                        gender=patient_gender,
                        age=patient_age
                    )
                hla_name = self._get_node_name(hla_action)
                is_forced = False

                dialog_history.append({
                    "turn": display_turn,
                    "role": "Debug_Log",
                    "content": f"【模型决策】HLA选中: {hla_name}"
                })

            if hla_action == self.doctor.DIAGNOSIS_NODE_ID:
                _, top1_prob, top3, pred_idx = self._predict_stats(true_idx)
                final_confidence = top1_prob
                pred_name = self.doctor.idx_to_disease_name.get(pred_idx, f"Unknown_{pred_idx}")
                true_name = case_data.get("diagnosis", f"Unknown_{true_idx}")

                if pred_idx == true_idx:
                    is_success = True
                    is_top3 = True
                elif true_idx in top3:
                    is_top3 = True

                prefix = "⚠️ [强制诊断]" if is_forced else "✅ [主动诊断]"
                dialog_history.append({
                    "turn": display_turn if not is_forced else display_turn - 1,
                    "role": "Doctor",
                    "content": f"{prefix} #结果# {pred_name} (Prob: {top1_prob:.2f})\n#真实诊断# {true_name}"
                })
                break

            elif self.doctor.is_exam_node(hla_action):
                if hla_action in asked_entities_ids:
                    display_turn += 1
                    curr_state = self.doctor.get_state_embedding(f"{history}\n重复检查:{hla_name}").detach()
                    continue

                self.doctor.mark_whole_node_as_asked(hla_action)
                asked_entities_ids.add(hla_action)

                short_name = str(hla_name).split("-")[-1].split("(")[0]
                question = self.doctor.generate_natural_question(hla_action, lla_entity=None, mode="eval")

                if not question or question.strip() == "":
                    question = f"(医生正在检查{short_name}...)"

                sim_input = question + "\n(Please strictly reply in Chinese / 请务必用纯中文回答此检查结果)"

                case_id = case_data.get("case_id", case_data.get("patient_id", "unknown"))
                action_key = f"exam_{hla_name}"
                nl_response = self._sim_step_cached(simulator, case_id, action_key, sim_input)

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

                dialog_history.append({
                    "turn": display_turn,
                    "role": "Debug_Log",
                    "content": f"🧠 【实时记忆更新】: {self._memory_json()}"
                })

                curr_state = self.doctor.get_state_embedding(f"{history}\n{hla_name}").detach()
                display_turn += 1

            else:
                entity, log_prob, value, entropy = self.doctor.select_lla_action(curr_state, hla_action)

                if not entity:
                    curr_state = self.doctor.get_state_embedding(f"{history}\n无有效实体").detach()
                    display_turn += 1
                    continue

                if entity in asked_entities_ids:
                    curr_state = self.doctor.get_state_embedding(f"{history}\n重复问诊:{entity}").detach()
                    display_turn += 1
                    continue

                asked_entities_ids.add(entity)

                try:
                    log_prob_val = float(log_prob.detach().cpu().item())
                except Exception:
                    log_prob_val = float(log_prob) if log_prob is not None else 0.0

                dialog_history.append({
                    "turn": display_turn,
                    "role": "Debug_Log",
                    "content": f"【LLA决策】实体: {entity}(LogProb: {log_prob_val:.4f})"
                })

                question = self.doctor.generate_natural_question(hla_action, entity, mode="eval")
                sim_input = question + "\n(Please strictly reply in Chinese / 请务必根据你的英文病历用纯中文回答)"

                case_id = case_data.get("case_id", case_data.get("patient_id", "unknown"))
                action_key = f"inquiry_{entity}"
                nl_response = self._sim_step_cached(simulator, case_id, action_key, sim_input)

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

                dialog_history.append({
                    "turn": display_turn,
                    "role": "Debug_Log",
                    "content": f"🧠 【实时记忆更新】: {self._memory_json()}"
                })

                curr_state = self.doctor.get_state_embedding(f"{history}\n{hla_name}").detach()
                display_turn += 1

        record = {
            "patient_id": case_data.get("case_id", case_data.get("patient_id", 0)),
            "result": "Success" if is_success else "Fail",
            "is_top3": bool(is_top3),
            "turns": min(display_turn - 1, self.args.max_turns),
            "confidence": final_confidence,
            "dialog_history": dialog_history,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return record

    def evaluate(self, cases):
        os.makedirs(os.path.dirname(self.args.output_path), exist_ok=True)

        top1 = 0
        top3 = 0
        total_turns = 0
        total_confidence = 0.0

        with open(self.args.output_path, "w", encoding="utf-8") as fout:
            for case in tqdm(cases, desc="Evaluating", ncols=120):
                record = self.evaluate_case(case)

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                top1 += 1 if record["result"] == "Success" else 0
                top3 += 1 if record["is_top3"] else 0
                total_turns += record["turns"]
                total_confidence += record.get("confidence", 0.0)

        self.sim_cache.save()

        n = len(cases)
        print("\n📊 Eval Finished")
        print(f"Top-1 Acc: {top1 / n:.2%}")
        print(f"Top-3 Acc: {top3 / n:.2%}")
        print(f"Avg Confidence: {total_confidence / n:.4f}")
        print(f"Avg Turns: {total_turns / n:.2f}")
        print(f"History saved to: {self.args.output_path}")


def load_eval_cases(args):
    if args.eval_path and os.path.exists(args.eval_path):
        data_path = args.eval_path
    else:
        data_path = os.path.join(
            project_root,
            "src", "data", "processed", args.data,
            f"{args.data}_cases_eval.json"
        )
        if not os.path.exists(data_path):
            data_path = os.path.join(
                project_root,
                "src", "data", "processed", args.data,
                f"{args.data}_cases_test.json"
            )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到 eval/test 数据文件: {data_path}")

    print(f"📂 Eval data: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    cases = [c for c in cases if "diagnosis_idx" in c]

    if args.max_cases > 0:
        cases = cases[:args.max_cases]

    return cases


if __name__ == "__main__":
    logging.getLogger("transformers").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="DLBCL")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--name", type=str, default="eval")

    parser.add_argument("--load_path", type=str, required=True, help="src/train/models/doctor_best_Qwen_only_s_v1.0.pth")
    parser.add_argument("--eval_path", type=str, default="", help="可选：指定 eval/test json")
    parser.add_argument("--output_path", type=str, default="", help="输出 JSONL history 路径")
    parser.add_argument("--max_cases", type=int, default=-1)

    parser.add_argument("--patient_model_path", type=str, default="/data/pretrained_models/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_path", type=str, default="/data/pretrained_models/GLM-4-9B-Chat",
                        help="Doctor Backbone 的模型路径")
    # 为兼容 DoctorAgent 初始化
    parser.add_argument("--eps_start", type=float, default=0.0)
    parser.add_argument("--eps_end", type=float, default=0.0)
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--diag_conf_thres", type=float, default=0.55)
    # 👇 新增这一行：接收模式参数
    parser.add_argument("--hla_mode", type=str, default="full",
                        choices=["pure_inquiry", "inquiry_plus_physical", "full"])

    args = parser.parse_args()

    if not args.output_path:
        out_dir = os.path.join(project_root, "logs", "eval")
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = os.path.join(out_dir, f"eval_history_{args.data}_{args.name}_{timestamp}.jsonl")

    cases = load_eval_cases(args)
    evaluator = Evaluator(args)
    evaluator.evaluate(cases)