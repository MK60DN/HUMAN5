"""
左脑适配器基类。
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class BaseAdapter(ABC):
    """所有左脑适配器的基类"""

    def __init__(
            self,
            domain: str = "general",
            adapter_name: Optional[str] = None,
            **kwargs
    ):
        """
        初始化适配器

        参数:
            domain: 适配领域
            adapter_name: 适配器名称
            **kwargs: 附加参数
        """
        self.domain = domain
        self.adapter_name = adapter_name or f"{self.__class__.__name__}_{domain}"
        self.is_trained = False
        self.logger = logging.getLogger(f"brainforge.adapter.{self.adapter_name}")

    @abstractmethod
    async def train_async(
            self,
            data: List[Dict[str, str]],
            epochs: int = 3,
            batch_size: int = 8,
            **kwargs
    ) -> Dict[str, Any]:
        """
        异步训练适配器

        参数:
            data: 训练数据
            epochs: 训练轮数
            batch_size: 批处理大小
            **kwargs: 附加训练参数

        返回:
            训练结果统计
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存适配器

        参数:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)

        # 保存适配器信息
        info = {
            "type": self.__class__.__name__.replace("Adapter", ""),
            "domain": self.domain,
            "name": self.adapter_name,
            "is_trained": self.is_trained
        }

        with open(os.path.join(path, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseAdapter":
        """
        加载适配器

        参数:
            path: 加载路径

        返回:
            适配器实例
        """
        pass