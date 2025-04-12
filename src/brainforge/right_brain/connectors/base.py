"""
API连接器基类。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class BaseConnector(ABC):
    """所有右脑API连接器的基类"""

    def __init__(
            self,
            api_key: str,
            model: str,
            **kwargs
    ):
        """
        初始化连接器

        参数:
            api_key: API密钥
            model: 模型标识符
            **kwargs: 附加参数
        """
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(f"brainforge.connector.{self.__class__.__name__}")

    @abstractmethod
    async def generate(
            self,
            prompt: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        调用API生成响应

        参数:
            prompt: 提示文本
            **kwargs: 生成参数

        返回:
            API响应
        """
        pass

    @abstractmethod
    def estimate_cost(
            self,
            prompt: str,
            response: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        估计API调用成本

        参数:
            prompt: 提示文本
            response: API响应(可选)

        返回:
            估计成本(美元)
        """
        pass