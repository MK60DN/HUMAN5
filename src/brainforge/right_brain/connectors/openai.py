"""
OpenAI API连接器实现。
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

import aiohttp
from openai import AsyncOpenAI, OpenAIError


class OpenAIConnector:
    """OpenAI API连接器"""

    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4",
            organization: Optional[str] = None,
            api_base: Optional[str] = None,
            max_retries: int = 3,
            timeout: float = 60.0,
            **kwargs
    ):
        """
        初始化OpenAI连接器

        参数:
            api_key: OpenAI API密钥
            model: 使用的模型名称
            organization: 组织ID(可选)
            api_base: API基础URL(可选)
            max_retries: 最大重试次数
            timeout: 超时时间(秒)
            **kwargs: 其他参数
        """
        self.api_key = api_key
        self.model = model
        self.organization = organization
        self.api_base = api_base
        self.max_retries = max_retries
        self.timeout = timeout

        self.logger = logging.getLogger("brainforge.connector.OpenAI")
        self.client = self._init_client()

        # 价格映射(每1K令牌价格，美元)
        self.price_map = {
            # 输入价格
            "input": {
                "gpt-4": 0.03,
                "gpt-4-turbo": 0.01,
                "gpt-3.5-turbo": 0.0015,
                "gpt-4o": 0.005
            },
            # 输出价格
            "output": {
                "gpt-4": 0.06,
                "gpt-4-turbo": 0.03,
                "gpt-3.5-turbo": 0.002,
                "gpt-4o": 0.015
            }
        }

    def _init_client(self) -> AsyncOpenAI:
        """初始化API客户端"""
        try:
            client_kwargs = {
                "api_key": self.api_key,
            }

            if self.organization:
                client_kwargs["organization"] = self.organization

            if self.api_base:
                client_kwargs["base_url"] = self.api_base

            client = AsyncOpenAI(**client_kwargs)
            self.logger.info(f"OpenAI client initialized for model: {self.model}")
            return client

        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用OpenAI API生成响应

        参数:
            prompt: 提示文本
            temperature: 温度(随机性)
            max_tokens: 最大生成令牌数
            top_p: 核采样
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            stop: 停止序列
            **kwargs: 其他参数

        返回:
            API响应
        """
        self.logger.debug(f"Generating with prompt: {prompt[:50]}...")

        # 构建API请求参数
        params = {
            "model": self.model,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if stop is not None:
            params["stop"] = stop

        # 添加其他参数
        params.update(kwargs)

        # 设置消息
        params["messages"] = [{"role": "user", "content": prompt}]

        # 尝试调用API
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"API attempt {attempt + 1}/{self.max_retries + 1}")

                # 调用API
                response = await self.client.chat.completions.create(**params)

                # 处理响应
                result = {
                    "id": response.id,
                    "model": response.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }

                # 提取生成的文本
                for choice in response.choices:
                    result["choices"].append({
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason,
                        "index": choice.index
                    })

                self.logger.debug(f"Generated text: {result['choices'][0]['message']['content'][:50]}...")
                return result

            except OpenAIError as e:
                self.logger.warning(f"OpenAI API error: {str(e)}")

                # 如果还有重试次数，则等待后重试
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # 指数退避
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error("Max retries exceeded")
                    raise

            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                raise

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
        try:
            # 如果有响应，使用实际token使用量
            if response and "usage" in response:
                prompt_tokens = response["usage"]["prompt_tokens"]
                completion_tokens = response["usage"]["completion_tokens"]
            else:
                # 估计token数量(每4个字符约1个token)
                prompt_tokens = len(prompt) // 4
                completion_tokens = 0  # 无法估计完成token

            # 查找价格
            model_key = self.model
            if model_key not in self.price_map["input"]:
                # 默认使用gpt-3.5-turbo价格
                model_key = "gpt-3.5-turbo"

            # 计算成本
            input_cost = (prompt_tokens / 1000) * self.price_map["input"][model_key]
            output_cost = (completion_tokens / 1000) * self.price_map["output"][model_key]
            total_cost = input_cost + output_cost

            return total_cost

        except Exception as e:
            self.logger.error(f"Error estimating cost: {str(e)}")
            return 0.0