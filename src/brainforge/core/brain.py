"""
BrainForge核心类，负责协调左右脑协作。
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio

from ..left_brain.adapters.base import BaseAdapter
from ..right_brain.connectors.base import BaseConnector
from .bridge import Bridge
from .config import BrainForgeConfig, DEFAULT_CONFIG


class BrainForge:
    """左右脑协作的高效LLM微调系统主类"""

    def __init__(
            self,
            right_brain: BaseConnector,
            left_brain: BaseAdapter,
            config: Optional[BrainForgeConfig] = None
    ):
        """
        初始化BrainForge系统

        参数:
            right_brain: API连接器实例
            left_brain: 适配器实例
            config: 系统配置
        """
        self.right_brain = right_brain
        self.left_brain = left_brain
        self.config = config or DEFAULT_CONFIG
        self.bridge = Bridge(self.left_brain, self.right_brain)

        # 设置日志
        self.logger = logging.getLogger("brainforge")
        self._setup_logging()

        self.logger.info(f"BrainForge initialized with {type(left_brain).__name__} "
                         f"and {type(right_brain).__name__}")

    def _setup_logging(self):
        """设置日志配置"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, self.config.log_level))

    async def _train_async(
            self,
            data_path: str,
            epochs: int = 3,
            batch_size: int = 8,
            **kwargs
    ) -> Dict[str, Any]:
        """异步训练左脑适配器"""
        self.logger.info(f"Starting training with {data_path}, epochs={epochs}, batch_size={batch_size}")

        # 加载数据
        from ..data.loader import load_data
        training_data = load_data(data_path)

        # 训练适配器
        train_results = await self.left_brain.train_async(
            training_data,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

        self.logger.info(f"Training completed: {train_results}")
        return train_results

    def train(
            self,
            data_path: str,
            epochs: int = 3,
            batch_size: int = 8,
            **kwargs
    ) -> Dict[str, Any]:
        """
        训练左脑适配器

        参数:
            data_path: 训练数据路径
            epochs: 训练轮数
            batch_size: 批处理大小
            **kwargs: 其他训练参数

        返回:
            训练结果统计
        """
        return asyncio.run(self._train_async(data_path, epochs, batch_size, **kwargs))

    async def _generate_async(self, prompt: str, **kwargs) -> str:
        """异步生成响应"""
        self.logger.debug(f"Processing input: {prompt[:50]}...")

        # 使用桥接层处理请求
        result = await self.bridge.process(prompt, **kwargs)

        return result["output"]

    def generate(self, prompt: str, **kwargs) -> str:
        """
        使用协作系统生成响应

        参数:
            prompt: 用户输入提示
            **kwargs: 生成参数

        返回:
            生成的响应文本
        """
        return asyncio.run(self._generate_async(prompt, **kwargs))

    async def _evaluate_async(
            self,
            test_data: Union[str, List[Dict[str, str]]],
            metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """异步评估系统性能"""
        self.logger.info("Starting evaluation")

        # 加载测试数据
        if isinstance(test_data, str):
            from ..data.loader import load_data
            test_items = load_data(test_data)
        else:
            test_items = test_data

        # 默认指标
        if metrics is None:
            metrics = ["accuracy", "relevance", "fluency"]

        # 加载评估器
        from ..evaluation.metrics import get_evaluators
        evaluators = get_evaluators(metrics)

        # 评估结果
        results = {metric: 0.0 for metric in metrics}
        total_items = len(test_items)

        # 对每个测试样本进行评估
        for i, item in enumerate(test_items):
            self.logger.debug(f"Evaluating item {i + 1}/{total_items}")

            # 生成响应
            generated = await self._generate_async(item["input"])

            # 计算每个指标
            for metric, evaluator in evaluators.items():
                score = evaluator(generated, item.get("reference", ""))
                results[metric] += score / total_items

        self.logger.info(f"Evaluation completed: {results}")
        return results

    def evaluate(
            self,
            test_data: Union[str, List[Dict[str, str]]],
            metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        评估系统性能

        参数:
            test_data: 测试数据路径或测试样本列表
            metrics: 要评估的指标列表

        返回:
            评估结果统计
        """
        return asyncio.run(self._evaluate_async(test_data, metrics))

    def save(self, path: str) -> None:
        """
        保存系统状态

        参数:
            path: 保存目录路径
        """
        self.logger.info(f"Saving BrainForge state to {path}")

        # 创建目录
        os.makedirs(path, exist_ok=True)

        # 保存左脑适配器
        adapter_path = os.path.join(path, "left_brain")
        self.left_brain.save(adapter_path)

        # 保存配置
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.dict(), f, indent=2)

        self.logger.info("Save completed")

    @classmethod
    def load(
            cls,
            path: str,
            right_brain: BaseConnector
    ) -> "BrainForge":
        """
        加载系统状态

        参数:
            path: 保存目录路径
            right_brain: API连接器实例

        返回:
            BrainForge实例
        """
        logger = logging.getLogger("brainforge")
        logger.info(f"Loading BrainForge state from {path}")

        # 加载配置
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            config = BrainForgeConfig(**config_dict)

        # 加载左脑适配器
        adapter_path = os.path.join(path, "left_brain")
        adapter_info_path = os.path.join(adapter_path, "info.json")

        with open(adapter_info_path, "r") as f:
            adapter_info = json.load(f)
            adapter_type = adapter_info["type"]

        # 动态导入适配器类
        from importlib import import_module
        adapter_module = import_module(f"..left_brain.adapters.{adapter_type.lower()}", package=__package__)
        adapter_class = getattr(adapter_module, f"{adapter_type}Adapter")

        # 创建适配器实例
        left_brain = adapter_class.load(adapter_path)

        # 创建BrainForge实例
        instance = cls(right_brain=right_brain, left_brain=left_brain, config=config)
        logger.info("Load completed")

        return instance