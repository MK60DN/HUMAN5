"""
医疗领域适配示例。
"""

import os
import logging
import asyncio
from dotenv import load_dotenv

from brainforge import BrainForge
from brainforge.left_brain.adapters import LoRAAdapter
from brainforge.right_brain.connectors import AnthropicConnector

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # 创建医疗领域适配器
    medical_adapter = LoRAAdapter(
        domain="medical",
        rank=16,
        alpha=32,
        dropout=0.05,
        base_model_name="facebook/opt-350m"  # 使用较小的模型进行本地适配
    )

    # 创建Claude API连接器
    claude_api = AnthropicConnector(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-opus-20240229"
    )

    # 创建BrainForge实例
    medical_assistant = BrainForge(
        left_brain=medical_adapter,
        right_brain=claude_api
    )

    # 示例训练数据
    training_data = [
        {
            "input": "What are the symptoms of pneumonia?",
            "output": "Pneumonia symptoms include fever, cough with phlegm, difficulty breathing, chest pain, fatigue, and sometimes confusion in older adults. Symptoms can vary from mild to severe depending on factors like the pathogen type, age, and overall health status."
        },
        {
            "input": "How is hypertension diagnosed?",
            "output": "Hypertension (high blood pressure) is diagnosed when blood pressure readings consistently show systolic pressure ≥130 mmHg and/or diastolic pressure ≥80 mmHg. Diagnosis requires multiple readings over time, as blood pressure naturally fluctuates. Home monitoring and 24-hour ambulatory monitoring may be used to confirm the diagnosis."
        },
        {
            "input": "What is the treatment for type 2 diabetes?",
            "output": "Type 2 diabetes treatment typically includes lifestyle modifications (diet, exercise, weight management) and medication. First-line pharmacotherapy is usually metformin. Additional medications may include sulfonylureas, GLP-1 receptor agonists, SGLT-2 inhibitors, DPP-4 inhibitors, thiazolidinediones, or insulin, depending on individual factors and glycemic control."
        }
    ]

    # 保存训练数据
    import json
    with open("medical_training.json", "w") as f:
        json.dump(training_data, f, indent=2)

    # 训练适配器
    logger.info("Training medical adapter...")
    results = await medical_assistant._train_async(
        data_path="medical_training.json",
        epochs=3,
        batch_size=1
    )
    logger.info(f"Training results: {results}")

    # 测试适配效果
    test_queries = [
        "What are the risk factors for heart disease?",
        "How should I manage my diabetes?",
        "What are the side effects of lisinopril?"
    ]

    logger.info("Testing medical assistant...")
    for query in test_queries:
        logger.info(f"Query: {query}")
        response = await medical_assistant._generate_async(query)
        logger.info(f"Response: {response[:100]}...")
        logger.info("-" * 50)

    # 保存模型
    os.makedirs("models", exist_ok=True)
    medical_assistant.save("models/medical_assistant")
    logger.info("Model saved to models/medical_assistant")


if __name__ == "__main__":
    asyncio.run(main())