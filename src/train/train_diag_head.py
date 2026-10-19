# -*- coding: utf-8 -*-
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import argparse


# ================= 0. 解析命令行参数 =================
def get_args():
    parser = argparse.ArgumentParser(description="Universal Medical Diagnosis Head Training")
    parser.add_argument("--data", type=str, required=True, help="数据集名称，如 IIYI 或 DLBCL")
    parser.add_argument("--model_path", type=str, required=True, help="LLM backbone 模型路径")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5, help="LoRA 的基准学习率")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"], help="池化方式")
    return parser.parse_args()


args = get_args()

# ================= 1. 配置参数与动态路径 =================
MAX_LEN = 768
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = 1#4
EPOCHS = args.epochs
LEARNING_RATE = args.lr  # 这里作为 LoRA 学习率
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
AUGMENTATION_FACTOR = 1
MASK_PROB = 0.08
SEED = 42

LLM_MODEL_PATH = args.model_path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_NAME = args.data
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data", "processed", DATASET_NAME)

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "Diagnosis head", f"cleaned_{DATASET_NAME}_cases_train.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "Diagnosis head", f"cleaned_{DATASET_NAME}_cases_test.json")
LABEL_JSON_PATH = os.path.join(DATA_DIR, "disease_labels.json")

SAVE_DIR = os.path.join(PROJECT_ROOT, "src", "model", "checkpoints")
model_folder_name = os.path.basename(LLM_MODEL_PATH.rstrip('/'))
SAVE_PATH = os.path.join(SAVE_DIR, f"diag_head_{model_folder_name}_{DATASET_NAME}_best.pth")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ================= 2. 数据处理 =================
class MedicalProcessor:
    def __init__(self, label_encoder):
        self.le = label_encoder

    def format_input(self, item, is_augment=False):
        info = item.get('cleaned_info', {})
        comp = info.get('主诉', '')
        pos = info.get('阳性', [])
        neg = info.get('阴性', [])
        if is_augment:
            if random.random() < 0.5: random.shuffle(pos)
            pos = [p for p in pos if random.random() > MASK_PROB]

        return f"医学诊断决策分析：\n[主诉情况]: {comp}\n[阳性体征/检查]: {', '.join(pos)}\n[阴性排除项]: {', '.join(neg)}\n请基于证据给出临床诊断："


class DiseaseDataset(Dataset):
    def __init__(self, data, tokenizer, processor, is_train=False):
        self.samples = []
        self.tokenizer = tokenizer
        for item in data:
            try:
                label = processor.le.transform([item['diagnosis']])[0]
                self.samples.append((processor.format_input(item, False), label))
                if is_train:
                    for _ in range(AUGMENTATION_FACTOR):
                        self.samples.append((processor.format_input(item, True), label))
            except:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(),
                'labels': torch.tensor(label)}


# ================= 3. 通用智能适配模型架构 =================
class UniversalMedicalClassifier(nn.Module):
    def __init__(self, model_path, num_labels, pooling_method="mean"):
        super().__init__()
        self.pooling_method = pooling_method

        path_lower = model_path.lower()
        if "glm" in path_lower:
            print(" -> 检测到 GLM 架构，正在配置 ChatGLM 专属 LoRA 目标层...")
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        else:
            print(" -> 检测到 Qwen/Llama 架构，正在配置标准 LoRA 目标层...")
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.base_model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=target_modules
        )
        self.llm = get_peft_model(self.base_model, peft_config)

        # 🔄 还原点：完全恢复你原本设计的 3 层全连接分类头结构，确保与 Qwen2.5 实验对齐
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_labels)
        )
        self.device = self.base_model.device
        self.classifier.to(self.device)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

        if self.pooling_method == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = last_hidden_state[:, -1, :]

        return self.classifier(pooled.float())


# ================= 4. 主循环 =================
def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    print(f"🚀 任务启动 | 数据集: {DATASET_NAME} | 基座模型: {model_folder_name} | 池化: {args.pooling}")

    with open(LABEL_JSON_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    ordered_class_names = [item[1] for item in sorted(label_map.items(), key=lambda x: int(x[0]))] if isinstance(
        label_map, dict) else label_map
    le = LabelEncoder()
    le.classes_ = np.array(ordered_class_names)

    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    train_labels_idx = le.transform([d['diagnosis'] for d in train_data if d['diagnosis'] in ordered_class_names])
    weights = compute_class_weight('balanced', classes=np.unique(train_labels_idx), y=train_labels_idx)
    class_weights = torch.ones(len(le.classes_)).cuda()
    for idx, w in zip(np.unique(train_labels_idx), weights): class_weights[idx] = w

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_dl = DataLoader(DiseaseDataset(train_data, tokenizer, MedicalProcessor(le), True), batch_size=BATCH_SIZE,
                          shuffle=True)
    test_dl = DataLoader(DiseaseDataset(test_data, tokenizer, MedicalProcessor(le), False), batch_size=BATCH_SIZE)

    model = UniversalMedicalClassifier(LLM_MODEL_PATH, len(le.classes_), pooling_method=args.pooling)

    # 🛠️ 核心优化：针对 3 层复杂分类头，使用分层学习率进行强力破局！
    # LoRA 权重用适合它的慢速微调率（2e-5），随机初始化的 3 层分类头用强力率（1e-3）
    optimizer = torch.optim.AdamW([
        {'params': model.llm.parameters(), 'lr': LEARNING_RATE},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0.0

        for step, batch in enumerate(pbar):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(logits, batch['labels'].cuda()) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 挂载 Loss 监控，让我们能看到 3 层网络每一步的收敛状态
            pbar.set_postfix(loss=f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")

        scheduler.step()

        # 验证阶段
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_dl:
                out = model(batch['input_ids'], batch['attention_mask'])
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                trues.extend(batch['labels'].numpy())

        acc = accuracy_score(trues, preds)
        print(f"📊 Epoch {epoch + 1} | Avg Train Loss: {epoch_loss / len(train_dl):.4f} | Test Acc: {acc:.4%}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🏆 发现更优诊断头，成功保存: {SAVE_PATH}")


if __name__ == "__main__":
    main()