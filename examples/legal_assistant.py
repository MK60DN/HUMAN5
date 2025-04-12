"""
法律领域适配示例。
"""

import os
import logging
import asyncio
from dotenv import load_dotenv

from brainforge import BrainForge
from brainforge.left_brain.adapters import PromptTuningAdapter
from brainforge.right_brain.connectors import OpenAIConnector

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # 创建法律领域适配器
    legal_adapter = PromptTuningAdapter(
        domain="legal",
        num_virtual_tokens=30,
        prompt_tuning_init="TEXT",
        prompt_text="As a legal professional, I provide accurate legal analysis: ",
        base_model_name="facebook/opt-350m"
    )

    # 创建OpenAI API连接器
    openai_api = OpenAIConnector(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )

    # 创建BrainForge实例
    legal_assistant = BrainForge(
        left_brain=legal_adapter,
        right_brain=openai_api
    )

    # 示例训练数据
    training_data = [
        {
            "input": "What constitutes a valid contract?",
            "output": "A valid contract requires offer, acceptance, consideration, legal capacity, legal purpose, and mutual assent (meeting of the minds). In some jurisdictions and for certain contract types, additional requirements such as writing (Statute of Frauds) may apply. The absence of any essential element may render the contract void, voidable, or unenforceable."
        },
        {
            "input": "Explain the concept of negligence in tort law.",
            "output": "Negligence in tort law consists of four elements: duty of care, breach of that duty, causation (both actual and proximate), and damages. The plaintiff must prove all elements to establish liability. Duty of care is determined by the reasonable person standard, which varies based on relationship, foreseeability of harm, and public policy considerations."
        },
        {
            "input": "What is the difference between a trademark and copyright?",
            "output": "Trademarks protect brand identifiers like names, logos, and slogans that distinguish goods/services in commerce, while copyrights protect original creative works (literature, music, art, etc.). Trademarks primarily prevent consumer confusion in the marketplace and last indefinitely with proper use and renewal. Copyrights protect creative expression and last for the author's life plus 70 years (in the US), without renewal requirements."
        }
    ]

    # 保存训练数据
    import json
    with open("legal_training.json", "w") as f:
        json.dump(training_data, f, indent=2)

    # 训练适配器
    logger.info("Training legal adapter...")
    results = await legal_assistant._train_async(
        data_path="legal_training.json",
        epochs=5,
        batch_size=1
    )
    logger.info(f"Training results: {results}")

    # 测试适配效果
    test_queries = [
        "What are the elements of a valid will?",
        "Explain the doctrine of res judicata.",
        "What constitutes intellectual property infringement?"
    ]

    logger.info("Testing legal assistant...")
    for query in test_queries:
        logger.info(f"Query: {query}")
        response = await legal_assistant._generate_async(query)
        logger.info(f"Response: {response[:100]}...")
        logger.info("-" * 50)

    # 保存模型
    os.makedirs("models", exist_ok=True)
    legal_assistant.save("models/legal_assistant")
    logger.info("Model saved to models/legal_assistant")


if __name__ == "__main__":
    asyncio.run(main())