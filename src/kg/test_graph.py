import torch
import os
import json
import sys

# 临时将 src 加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.doctor.rl_LLM import DoctorAgent


def test_graph_head():
    print("🧪 开始测试 Graph Diagnosis Head...")

    # =======================================================
    # 🟢 修改这里：指向你的 disease_labels.json
    # =======================================================
    label_path = os.path.join(project_root, "data", "processed", "disease_labels.json")

    # 检查文件是否存在
    if not os.path.exists(label_path):
        print(f"❌ 错误: 找不到标签映射文件: {label_path}")
        return

    print(f"📄 加载标签文件: {label_path}")

    # 1. 初始化 Doctor (传入 label_path)
    # 代码会自动处理 (J20.8) 这种后缀，也会自动反转字典
    doctor = DoctorAgent(label_path=label_path)

    # 2. 模拟一个对话状态
    fake_state = torch.randn(2, doctor.state_dim).to(doctor.device)

    # 3. 尝试预测
    print("🔮 正在进行前向传播 (Forward Pass)...")
    logits = doctor.predict_diagnosis(fake_state)

    # 4. 检查输出
    print(f"✅ 输出形状: {logits.shape}")

    # 5. 检查是否真的加载了图
    if doctor.adj_matrix is not None:
        edge_count = doctor.adj_matrix.nonzero().size(0)
        print(f"📊 邻接矩阵中的边数量: {edge_count}")

        if edge_count == 0:
            print("⚠️ 警告: 边数量仍为 0。请检查 CSV 里的名字和 JSON 里的名字是否匹配。")
            print("注意：JSON 里的 '细菌性支气管肺炎 (J20.8)' 会被代码自动清洗为 '细菌性支气管肺炎'")
            print("请确认 disease_kg_entity.csv 里是否有 '细菌性支气管肺炎' 这个实体。")
        else:
            print(f"🎉 成功: 图结构已注入模型！成功建立了 {edge_count} 条疾病-症状关联。")


if __name__ == "__main__":
    test_graph_head()
