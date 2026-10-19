import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class PatientSimulator:
    def __init__(self,
                 model_path="/data/pretrained_models/Qwen2.5-7B-Instruct",
                 patient_data=None,
                 model=None,
                 tokenizer=None,
                 device=None):
        """
        统一配置驱动的患者模拟器
        :param patient_data: 支持你提供的统一 JSON 结构
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载模型与分词器
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            print(f"🚀 正在初始化模型: {os.path.basename(model_path)} (4bit 量化)...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # ==================== 4bit 量化配置 ====================
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            # ============================================================

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            ).eval()  # device_map="auto" 已自动处理设备，无需额外 .to()

        # 2. 初始化数据与状态
        self.patient_data = patient_data or self._get_default_patient_data()
        self.system_prompt = self._build_system_prompt()
        self.history = []

    # 其余方法保持完全不变（_get_default_patient_data、_detect_language、_build_system_prompt、reset、step）
    def _get_default_patient_data(self):
        """默认使用你提供的英文数据集样本作为缺省值"""
        return {
            "case_id": 1002,
            "raw_medical_record": {
                "一般资料": "Sex: F, Service: MEDICINE",
                "主诉": "Cough",
                "现病史": "Patient developed a cough 4 days prior... no fever... No chest pain.",
                "既往史": "Breast CA x 3, NSCLC, HTN, PE",
                "查体": "VITALS: BP: 138/84 HR: 84. LUNGS: bilateral rhonchi.",
                "辅助检查": "Hgb: 7.4, K: 2.7. CT Chest: patchy bronchovascular opacities."
            }
        }

    def _detect_language(self, text):
        """根据内容字符占比自动识别语种"""
        if not text: return "zh"
        en_chars = len(re.findall(r'[a-zA-Z]', str(text)))
        return "en" if en_chars > (len(str(text)) * 0.3) else "zh"

    def _build_system_prompt(self):
        """
        配置驱动的核心：根据 JSON 内容动态生成 Prompt。
        已强化空值防护，避免 NoneType 拼接错误。
        """
        # --- 1. 安全提取 patient_data ---
        data = self.patient_data
        if isinstance(data, dict) and "raw_medical_record" in data:
            record = data["raw_medical_record"]
        else:
            record = data if isinstance(data, dict) else {}

        # --- 2. 字段安全提取 + None 转字符串 ---
        info = record.get('一般资料') or record.get('general_info') or "未知"
        cc = record.get('主诉') or record.get('chief_complaint') or "无"
        hpi = record.get('现病史') or record.get('history_of_present_illness') or "无"
        pmh = record.get('既往史') or record.get('past_medical_history') or "无"
        pe = record.get('查体') or record.get('physical_examination') or "无"
        aux = record.get('辅助检查') or record.get('auxiliary_examination') or "无"

        # 强制转为字符串（彻底防止 None）
        info = str(info) if info is not None else "未知"
        cc = str(cc) if cc is not None else "无"
        hpi = str(hpi) if hpi is not None else "无"
        pmh = str(pmh) if pmh is not None else "无"
        pe = str(pe) if pe is not None else "无"
        aux = str(aux) if aux is not None else "无"

        # --- 3. 语种检测（安全拼接）---
        text_for_lang = hpi + cc
        is_english = self._detect_language(text_for_lang) == "en"
        translation_guide = "【重要：后台病历为英文，你必须将其翻译并总结为专业中文回答】" if is_english else ""

        prompt = f"""你现在是一个医疗对话模拟系统，包含两个身份：【患者】和【医疗环境】。
    {translation_guide} 你的所有输出必须【全部使用简体中文】。

    【后台病历信息】
    - 基本信息: {info}
    - 主诉: {cc}
    - 现病史: {hpi}
    - 既往史: {pmh}
    - 体格检查: {pe}
    - 辅助检查: {aux}

    【交互准则】
    1. 【日常问诊 (患者身份)】：
       - 如果医生问“你怎么了/哪里不舒服”，你**只能**回答最核心主诉部分的 1-2 个症状（例如：“我腹股沟两边疼”），一句话以内。
       - 绝对不要主动添加持续时间、既往病史或其他未被询问的内容。
       - 医生问什么，你只回答什么，绝不主动背诵完整病历，绝不剧透诊断。

    2. 【客观检查 (医疗环境身份)】：
       - 当医生的指令包含查体或看报告动作（如：量血压、听诊、血常规、CT、骨穿、活检等）时，必须立即切换身份。
       - 直接输出对应的检查结果（英文请翻译为规范中文术语）。严禁输出“好的”、“医生你看”等对话废话。
       - 若病历未提及该检查，统一回复：“未见该项检查记录”。

    【正确与错误示例】
        医生说：“好，我给你量个生命体征。”
        ❌ 错误：好的医生，我的血压是...
        ✅ 正确：血压 138/84 mmHg，心率 84 次/分，血氧饱和度 100%。

        医生说：“给你开个血常规。”
        ❌ 错误：好的，请看我的化验单。
        ✅ 正确：血常规结果：白细胞 6.5×10^9/L，血红蛋白 7.4 g/dL（偏低）...

    【严格要求】
    - 所有回答必须简洁、专业、只基于后台病历。
    - 禁止 hallucination（幻觉），没有的信息不要编造。
    """

        return prompt

    def reset(self, patient_data=None):
        if patient_data:
            self.patient_data = patient_data
            self.system_prompt = self._build_system_prompt()
        self.history = []
        return "模拟器重置完毕。"

    def step(self, doctor_utterance: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": doctor_utterance})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.01,
                top_p=0.85,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.history.append({"role": "user", "content": doctor_utterance})
        self.history.append({"role": "assistant", "content": response.strip()})
        return response.strip()

# ================= 测试脚本 (保留你所有的原始用例) =================
if __name__ == "__main__":
    simulator = PatientSimulator(model_path="/data/pretrained_models/Qwen2.5-7B-Instruct")

    # # 1. 测试用例：英文病历数据集逻辑
    # print("\n--- [测试] 英文病历驱动场景 ---")
    # en_case = {
    #     "case_id": 1155,
    #     "raw_medical_record": {
    #             "一般资料": "Sex: F, Service: MEDICINE",
    #             "主诉": "Cough",
    #             "现病史": "___ PMH of PE (on warfarin), Breast CA x 3 (most recently Stage IIb left breast cancer, infiltrating carcinoma, ER+/PR+/HER 2 amplified, s/p C3 TCHP), prior NSCLC Stage IIIA (s/p neoadjuvant ___ x 2, surgery, chemotherapy and xrt), presents from clinic with cough\n\nAs per review of notes, patient last had surveillance PET imaging for hx of lung cancer in ___, which identified suspicious left axillary node right paratracheal node for which she had EBUS with IP, sampling negative for malignancy, so patient planned to repeat CT Torso in ___ prior to next oncology appointment. However, in breast ___ clinic today, pt reported cough, CXR revealed ___ opacity concerning for pneumonia so was referred to ED\n\nPt reports that she developed a cough 4 days prior to admission which is newly productive of clear/white sputum, small amounts. She noted that she does not have any SOB, fever, chills, sore throat, headache, rash or sick contacts. Noted that she has lost several pounds since starting chemotherapy, attributes it to loosing her taste for food. No chest pain/discomfort.",
    #             "既往史": "PAST ONCOLOGIC HISTORY: \n-Breast CA x 3\n-NSCLC Stage IIIA\n-Pulmonary Embolism, + lupus antiocoagulant, PE treated with lovenox now on warfarin \n-Glaucoma \n-HTN (hypertension) \n-Rheumatoid arthritis",
    #             "查体": "VITALS: ___ 1444 BP: 138/84 HR: 84 RR: 18 O2 sat: 100% O2 delivery: RA \nGENERAL: Sitting upright in bed, appears comfortable, no acute distress\nLUNGS: bilateral rhonchi, normal respiratory rate, no increased work of breathing, minimal cough, no egophony, increased tactile fremitus on L\nCV: RRR, +S1/S2, no m/r/g\nABD: Soft, NTND, +BS\nSKIN: Warm dry, no rash\nNEURO: Alert and oriented x3, fluent speech",
    #             "辅助检查": "ADMISSION LABS: \n___ 02:00PM BLOOD WBC: 6.5 RBC: 2.43* Hgb: 7.4* Hct: 23.7*\n___ 02:00PM BLOOD Glucose: 101* UreaN: 18 Creat: 0.8 Na: 144 K: 2.7* Cl: 105 HCO3: 27 AnGap: 12\n\nImaging:\nCT Chest (___):\nFINDINGS: \nThere are patchy new inflammatory changes in the right lung... mostly characterized by mucous plugging and opacities suggesting infection or inflammation of lower airways. In the left lower lobe, there are similar opacities but more confluent than on the right, including patchy bronchovascular opacities highly suggestive of pneumonia.\nIMPRESSION: \n1. Multifocal opacities consistent with fairly widespread bronchopneumonia."
    #         }
    # }
    # simulator.reset(en_case)
    # test_queries_en = [
    #     "您好，请问哪里不舒服？",
    #     "咳嗽多久了？有痰吗？",
    #     "最近有发烧、胸痛或者觉得喘不上气吗？",
    #     "以前得过什么重大疾病吗？比如肿瘤或者高血压？",
    #     "好，我给你量个生命体征（血压、心率、血氧）。",
    #     "我用听诊器听一下你的肺部。",
    #     "给你开个血常规和生化检查，看看结果。",  # 期望提取 Hgb: 7.3, K: 2.7
    #     "你的胸部CT结果出来了，我看一下报告。"
    # ]
    # for query in test_queries_en:
    #     print(f"\n👨‍⚕️ 医生: {query}")
    #     print(f"🤖 模拟器: {simulator.step(query)}")

    # 2. 测试用例：中文病历数据集逻辑
    print("\n--- [测试] 中文病历驱动场景 ---")
    cn_case = {
        "case_id": 1057,
        "raw_medical_record": {
            "一般资料": "女，40岁",
            "主诉": "反复咳嗽、咳痰伴气喘5年，加重3天。",
            "现病史": "患者5年前受凉后出现咳嗽、咳白粘痰，伴有活动后气喘。3天前受凉后上述症状加重，咳黄脓痰，夜间无法平卧。",
            "既往史": "否认高血压、糖尿病史。有吸烟史20年，每天1包。",
            "查体": "T 37.8℃，P 98次/分，R 24次/分，BP 130/80mmHg。神志清，精神可。桶状胸，双肺呼吸音减弱，可闻及散在干湿性啰音。",
            "辅助检查": "血常规：WBC 11.2x10^9/L，N 82%。胸部CT：双肺纹理增粗、紊乱，可见肺气肿征象。"
        }
    }
    simulator.reset(cn_case)
    test_queries_cn = [
        "你好，我是你的主治医生，你今天哪里不舒服？",
        "这个情况有多久了？",
        "平时抽烟吗？",
        "我给你量个血压，测个体温。",
        "听诊一下你的肺部。",
        "看看血常规结果出来了没。",
        "做个腹部B超。"
    ]
    for query in test_queries_cn:
        print(f"\n👨‍⚕️ 医生: {query}")
        print(f"🤒 模拟器: {simulator.step(query)}")