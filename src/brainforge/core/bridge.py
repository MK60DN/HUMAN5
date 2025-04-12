"""
BrainForge桥接层，负责协调左脑和右脑的交互。
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Union

from ..left_brain.adapters.base import BaseAdapter
from ..right_brain.connectors.base import BaseConnector


class Bridge:
    """
    大脑桥接层 - 负责协调左脑适配器和右脑API连接器的交互

    主要功能:
    1. 协调左右脑模块的工作流
    2. 处理前向传递和后向传递
    3. 收集反馈和性能指标用于优化
    """

    def __init__(
            self,
            left_brain: BaseAdapter,
            right_brain: BaseConnector,
            cache_enabled: bool = True,
            feedback_collection: bool = True,
            track_metrics: bool = True
    ):
        """
        初始化桥接层

        参数:
            left_brain: 左脑适配器实例
            right_brain: 右脑API连接器实例
            cache_enabled: 是否启用缓存
            feedback_collection: 是否收集反馈
            track_metrics: 是否跟踪性能指标
        """
        self.left_brain = left_brain
        self.right_brain = right_brain
        self.cache_enabled = cache_enabled
        self.feedback_collection = feedback_collection
        self.track_metrics = track_metrics

        self.cache = {}
        self.metrics = {
            "api_calls": 0,
            "tokens_used": 0,
            "avg_latency": 0,
            "cache_hits": 0,
            "errors": 0
        }

        self.logger = logging.getLogger("brainforge.bridge")

    def _generate_cache_key(self, input_text: str, kwargs: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 使用输入和参数的哈希作为缓存键
        key_data = input_text + json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_metrics(self, result: Dict[str, Any]) -> None:
        """更新性能指标"""
        self.metrics["api_calls"] += 1

        # 更新令牌使用量(如果可用)
        if "tokens" in result.get("metadata", {}):
            self.metrics["tokens_used"] += result["metadata"]["tokens"]

        # 更新平均延迟
        latency = result.get("metadata", {}).get("latency", 0)
        n = self.metrics["api_calls"]
        self.metrics["avg_latency"] = (self.metrics["avg_latency"] * (n - 1) + latency) / n

    async def _left_brain_preprocessing(
            self,
            input_text: str,
            **kwargs
    ) -> str:
        """左脑预处理输入"""
        self.logger.debug("Preprocessing input with left brain")
        return await self.left_brain.process_input(input_text, **kwargs)

    async def _right_brain_call(
            self,
            processed_input: str,
            **kwargs
    ) -> Dict[str, Any]:
        """调用右脑API"""
        self.logger.debug("Calling right brain API")
        return await self.right_brain.generate(processed_input, **kwargs)

    async def _left_brain_postprocessing(
            self,
            api_response: Dict[str, Any],
            original_input: str,
            **kwargs
    ) -> str:
        """左脑后处理输出"""
        self.logger.debug("Postprocessing output with left brain")
        return await self.left_brain.process_output(api_response, original_input, **kwargs)

    async def process(
            self,
            input_text: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        处理输入文本并返回结果

        参数:
            input_text: 用户输入文本
            **kwargs: 附加参数

        返回:
            包含处理结果和元数据的字典
        """
        # 记录起始时间以计算延迟
        start_time = time.time()

        # 搜索缓存
        cache_key = self._generate_cache_key(input_text, kwargs)
        if self.cache_enabled and cache_key in self.cache:
            self.metrics["cache_hits"] += 1
            self.logger.info(f"Cache hit for input: {input_text[:30]}...")
            return self.cache[cache_key]

        try:
            # 1. 左脑预处理输入
            processed_input = await self._left_brain_preprocessing(input_text, **kwargs)

            # 2. 调用右脑API
            api_response = await self._right_brain_call(processed_input, **kwargs)

            # 3. 左脑后处理输出
            processed_output = await self._left_brain_postprocessing(api_response, input_text, **kwargs)

            # 构建结果
            result = {
                "input": input_text,
                "processed_input": processed_input,
                "api_response": api_response,
                "output": processed_output,
                "metadata": {
                    "latency": time.time() - start_time,
                    "model": getattr(self.right_brain, "model", "unknown"),
                    "cache_hit": False,
                    "error": None
                }
            }

            # 更新缓存
            if self.cache_enabled:
                self.cache[cache_key] = result

            # 更新指标
            if self.track_metrics:
                self._update_metrics(result)

            return result

        except Exception as e:
            self.logger.error(f"Error in bridge processing: {str(e)}")
            self.metrics["errors"] += 1

            # 返回错误信息
            return {
                "input": input_text,
                "output": None,
                "metadata": {
                    "latency": time.time() - start_time,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            }