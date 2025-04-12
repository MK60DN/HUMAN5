"""
LoRA (Low-Rank Adaptation) 适配器实现。
"""

import os
import json
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from .base import BaseAdapter


class LoRAAdapter(BaseAdapter):
    """使用LoRA技术的参数高效微调适配器"""

    def __init__(
            self,
            domain: str = "general",
            rank: int = 8,
            alpha: int = 16,
            dropout: float = 0.1,
            base_model_name: str = "facebook/opt-125m",  # 轻量级模型作为本地适配器
            adapter_name: Optional[str] = None,
            **kwargs
    ):
        """
        初始化LoRA适配器

        参数:
            domain: 适配领域
            rank: LoRA秩
            alpha: LoRA缩放因子
            dropout: Dropout率
            base_model_name: 基础模型名称
            adapter_name: 适配器名称
            **kwargs: 其他参数
        """
        super().__init__(domain=domain, adapter_name=adapter_name, **kwargs)

        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.base_model_name = base_model_name

        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化底层模型"""
        try:
            self.logger.info(f"Loading base model: {self.base_model_name}")

            # 加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # 配置LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                target_modules=["q_proj", "v_proj"]  # 目标模块根据模型架构调整
            )

            # 将模型转换为PEFT模型
            self.model = get_peft_model(self.model, lora_config)

            self.logger.info("Model initialization completed")

        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise

        async def train_async(
                self,
                data: List[Dict[str, str]],
                epochs: int = 3,
                batch_size: int = 8,
                learning_rate: float = 3e-4,
                **kwargs
        ) -> Dict[str, Any]:
            """
            异步训练适配器

            参数:
                data: 训练数据列表，每项包含"input"和"output"
                epochs: 训练轮数
                batch_size: 批处理大小
                learning_rate: 学习率
                **kwargs: 其他训练参数

            返回:
                训练结果统计
            """
            self.logger.info(f"Starting LoRA training: {len(data)} samples, {epochs} epochs")

            # 异步训练实际是在另一个线程中同步运行
            def train_fn():
                try:
                    from torch.utils.data import Dataset, DataLoader
                    from transformers import Trainer, TrainingArguments

                    # 准备数据集
                    class TextDataset(Dataset):
                        def __init__(self, data, tokenizer):
                            self.inputs = []
                            self.labels = []

                            for item in data:
                                # 构建提示模板
                                template = f"Input: {item['input']}\nOutput: {item['output']}"

                                # 分词并添加特殊标记
                                encodings = tokenizer(
                                    template,
                                    truncation=True,
                                    max_length=512,
                                    padding="max_length",
                                    return_tensors="pt"
                                )

                                self.inputs.append(encodings["input_ids"][0])
                                self.labels.append(encodings["input_ids"][0].clone())

                                # 将非输出部分的标签设为-100
                                output_text = f"Output: {item['output']}"
                                output_encoding = tokenizer(output_text, add_special_tokens=False)
                                output_len = len(output_encoding["input_ids"])

                                input_len = len(self.labels[-1])
                                output_start = input_len - output_len

                                self.labels[-1][:output_start] = -100

                        def __len__(self):
                            return len(self.inputs)

                        def __getitem__(self, idx):
                            return {
                                "input_ids": self.inputs[idx],
                                "labels": self.labels[idx]
                            }

                    # 创建数据集
                    train_dataset = TextDataset(data, self.tokenizer)

                    # 设置训练参数
                    training_args = TrainingArguments(
                        output_dir="./results",
                        num_train_epochs=epochs,
                        per_device_train_batch_size=batch_size,
                        learning_rate=learning_rate,
                        weight_decay=0.01,
                        logging_dir="./logs",
                        logging_steps=10,
                        save_strategy="epoch",
                        remove_unused_columns=False,
                    )

                    # 创建训练器
                    trainer = Trainer(
                        model=self.model,
                        args=training_args,
                        train_dataset=train_dataset,
                    )

                    # 开始训练
                    train_result = trainer.train()

                    # 设置训练完成标志
                    self.is_trained = True

                    # 返回训练结果
                    return {
                        "train_loss": float(train_result.training_loss),
                        "train_runtime": train_result.metrics.get("train_runtime", 0),
                        "samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                        "epochs": epochs,
                        "trained_samples": len(data)
                    }

                except Exception as e:
                    self.logger.error(f"Training error: {str(e)}")
                    return {"error": str(e)}

            # 在事件循环的执行器中运行训练函数
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, train_fn)

            self.logger.info(f"Training completed: {result}")
            return result

        async def process_input(
                self,
                raw_input: str,
                **kwargs
        ) -> str:
            """
            处理输入文本

            参数:
                raw_input: 原始输入文本
                **kwargs: 附加参数

            返回:
                处理后的输入文本
            """
            # 如果模型未训练，则直接返回原始输入
            if not self.is_trained:
                self.logger.warning("Model not trained yet, returning raw input")
                return raw_input

            try:
                # 模板构建
                processed_input = raw_input

                # 这里可以根据需要添加领域特定的处理逻辑
                # 例如添加特定提示词、格式化输入等

                if self.domain == "medical":
                    processed_input = f"As a medical professional, please analyze: {raw_input}"
                elif self.domain == "legal":
                    processed_input = f"From a legal perspective, please answer: {raw_input}"
                elif self.domain == "technical":
                    processed_input = f"As a technical expert, please explain: {raw_input}"

                self.logger.debug(f"Processed input: {processed_input[:50]}...")
                return processed_input

            except Exception as e:
                self.logger.error(f"Error processing input: {str(e)}")
                return raw_input

        async def process_output(
                self,
                api_output: Dict[str, Any],
                original_input: str,
                **kwargs
        ) -> str:
            """
            处理API输出

            参数:
                api_output: API输出
                original_input: 原始输入文本
                **kwargs: 附加参数

            返回:
                处理后的输出文本
            """
            try:
                # 提取API响应文本
                if isinstance(api_output, dict):
                    api_text = api_output.get("text", "")
                    if not api_text and "choices" in api_output:
                        api_text = api_output["choices"][0].get("message", {}).get("content", "")
                    if not api_text and "generations" in api_output:
                        api_text = api_output["generations"][0].get("text", "")
                else:
                    api_text = str(api_output)

                # 如果模型未训练，则直接返回API输出
                if not self.is_trained:
                    self.logger.warning("Model not trained yet, returning API output directly")
                    return api_text

                # 后处理逻辑
                # 可以根据领域添加特定的后处理逻辑
                processed_output = api_text

                # 示例：针对不同领域进行调整
                if self.domain == "medical":
                    # 医学领域的专业术语处理
                    # 这里可以添加医学术语转换、格式规范化等
                    pass
                elif self.domain == "legal":
                    # 法律文本后处理
                    # 可以添加法律术语标准化、引用格式调整等
                    pass

                self.logger.debug(f"Processed output: {processed_output[:50]}...")
                return processed_output

            except Exception as e:
                self.logger.error(f"Error processing output: {str(e)}")
                # 出错时返回原始API输出
                if isinstance(api_output, dict) and "text" in api_output:
                    return api_output["text"]
                elif isinstance(api_output, str):
                    return api_output
                else:
                    return str(api_output)

        def save(self, path: str) -> None:
            """
            保存适配器

            参数:
                path: 保存路径
            """
            # 调用父类方法保存基本信息
            super().save(path)

            try:
                self.logger.info(f"Saving LoRA adapter to {path}")

                # 保存模型
                if self.is_trained and self.model is not None:
                    model_path = os.path.join(path, "model")
                    self.model.save_pretrained(model_path)
                    self.tokenizer.save_pretrained(model_path)

                # 保存配置
                config = {
                    "rank": self.rank,
                    "alpha": self.alpha,
                    "dropout": self.dropout,
                    "base_model_name": self.base_model_name,
                    "domain": self.domain
                }

                config_path = os.path.join(path, "config.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                self.logger.info(f"LoRA adapter saved successfully")

            except Exception as e:
                self.logger.error(f"Error saving adapter: {str(e)}")
                raise

        @classmethod
        def load(cls, path: str) -> "LoRAAdapter":
            """
            加载适配器

            参数:
                path: 加载路径

            返回:
                LoRAAdapter实例
            """
            logger = logging.getLogger(f"brainforge.adapter.LoRA")
            logger.info(f"Loading LoRA adapter from {path}")

            try:
                # 加载配置
                config_path = os.path.join(path, "config.json")
                with open(config_path, "r") as f:
                    config = json.load(f)

                # 创建适配器实例
                adapter = cls(
                    domain=config["domain"],
                    rank=config["rank"],
                    alpha=config["alpha"],
                    dropout=config["dropout"],
                    base_model_name=config["base_model_name"]
                )

                # 加载模型(如果存在)
                model_path = os.path.join(path, "model")
                if os.path.exists(model_path):
                    # 替换自动初始化的模型
                    adapter.model = None
                    adapter.tokenizer = None

                    # 加载保存的模型
                    adapter.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    adapter.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )

                    # 设置训练标志
                    adapter.is_trained = True
                    logger.info("Model loaded successfully")

                return adapter

            except Exception as e:
                logger.error(f"Error loading adapter: {str(e)}")
                raise