"""
API连接器单元测试。
"""

import pytest
import os
import json
from unittest.mock import AsyncMock, patch, MagicMock

from brainforge.right_brain.connectors.openai import OpenAIConnector
from brainforge.right_brain.connectors.anthropic import AnthropicConnector


# 测试OpenAI连接器
@pytest.mark.asyncio
async def test_openai_connector_init():
    """测试OpenAI连接器初始化"""
    with patch('openai.AsyncOpenAI') as mock_client:
        # 创建连接器
        connector = OpenAIConnector(
            api_key="test_key",
            model="gpt-4"
        )

        # 验证初始化
        assert connector.api_key == "test_key"
        assert connector.model == "gpt-4"
        assert mock_client.called


@pytest.mark.asyncio
async def test_openai_connector_generate():
    """测试OpenAI连接器生成功能"""
    with patch('openai.AsyncOpenAI') as mock_client:
        # 模拟OpenAI响应
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_choice = MagicMock()
        mock_choice.message.role = "assistant"
        mock_choice.message.content = "Generated response"
        mock_choice.finish_reason = "stop"
        mock_choice.index = 0

        mock_response.choices = [mock_choice]

        # 设置模拟客户端
        mock_client_instance = AsyncMock()
        mock_chat = AsyncMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_instance.chat = mock_chat
        mock_client.return_value = mock_client_instance

        # 创建连接器
        connector = OpenAIConnector(
            api_key="test_key",
            model="gpt-4"
        )

        # 测试生成方法
        result = await connector.generate("Test prompt")

        # 验证结果
        assert result["id"] == "resp_123"
        assert result["model"] == "gpt-4"
        assert result["choices"][0]["message"]["content"] == "Generated response"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_openai_connector_retry():
    """测试OpenAI连接器重试机制"""
    with patch('openai.AsyncOpenAI') as mock_client:
        from openai.types.chat.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
        from openai import OpenAIError

        # 设置模拟客户端
        mock_client_instance = AsyncMock()
        mock_chat = AsyncMock()

        # 第一次调用失败，第二次成功
        mock_chat.completions.create = AsyncMock(side_effect=[
            OpenAIError("Rate limit exceeded"),
            MagicMock()  # 成功响应
        ])

        mock_client_instance.chat = mock_chat
        mock_client.return_value = mock_client_instance

        # 创建连接器
        connector = OpenAIConnector(
            api_key="test_key",
            model="gpt-4",
            max_retries=1
        )

        # 测试重试机制
        with patch('asyncio.sleep', return_value=None):  # 跳过等待
            try:
                result = await connector.generate("Test prompt")
                # 如果成功执行到这里，说明重试成功
                assert True
            except Exception:
                pytest.fail("重试机制失败")


def test_openai_connector_cost_estimation():
    """测试OpenAI连接器成本估算"""
    # 创建连接器
    connector = OpenAIConnector(
        api_key="test_key",
        model="gpt-4"
    )

    # 测试有响应的情况
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50
        }
    }

    cost = connector.estimate_cost("Test prompt", response)
    expected_cost = (100 / 1000) * 0.03 + (50 / 1000) * 0.06
    assert cost == pytest.approx(expected_cost)

    # 测试无响应的情况
    cost = connector.estimate_cost("This is a test prompt with about twenty tokens")
    assert cost > 0  # 应该有估计成本

    # 测试未知模型
    connector.model = "unknown-model"
    cost = connector.estimate_cost("Test prompt", response)
    expected_cost = (100 / 1000) * 0.0015 + (50 / 1000) * 0.002  # 默认使用gpt-3.5-turbo价格
    assert cost == pytest.approx(expected_cost)


# 测试Anthropic连接器
@pytest.mark.asyncio
async def test_anthropic_connector_init():
    """测试Anthropic连接器初始化"""
    with patch('anthropic.AsyncAnthropic') as mock_client:
        # 创建连接器
        connector = AnthropicConnector(
            api_key="test_key",
            model="claude-3-opus-20240229"
        )

        # 验证初始化
        assert connector.api_key == "test_key"
        assert connector.model == "claude-3-opus-20240229"
        assert mock_client.called


@pytest.mark.asyncio
async def test_anthropic_connector_generate():
    """测试Anthropic连接器生成功能"""
    with patch('anthropic.AsyncAnthropic') as mock_client:
        # 模拟Anthropic响应
        mock_response = MagicMock()
        mock_response.id = "msg_123"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.content = [MagicMock(text="Generated response")]
        mock_response.stop_reason = "end_turn"
        mock_response.stop_sequence = None
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        # 设置模拟客户端
        mock_client_instance = AsyncMock()
        mock_messages = AsyncMock()
        mock_messages.create = AsyncMock(return_value=mock_response)
        mock_client_instance.messages = mock_messages
        mock_client.return_value = mock_client_instance

        # 创建连接器
        connector = AnthropicConnector(
            api_key="test_key",
            model="claude-3-opus-20240229"
        )

        # 测试生成方法
        result = await connector.generate("Test prompt")

        # 验证结果
        assert result["id"] == "msg_123"
        assert result["model"] == "claude-3-opus-20240229"
        assert result["content"] == "Generated response"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20


def test_anthropic_connector_cost_estimation():
    """测试Anthropic连接器成本估算"""
    # 创建连接器
    connector = AnthropicConnector(
        api_key="test_key",
        model="claude-3-opus-20240229"
    )

    # 测试有响应的情况
    response = {
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500
        }
    }

    cost = connector.estimate_cost("Test prompt", response)
    expected_cost = (1000 / 1000000) * 15.0 + (500 / 1000000) * 75.0
    assert cost == pytest.approx(expected_cost)

    # 测试无响应的情况
    cost = connector.estimate_cost("This is a test prompt with about twenty tokens")
    assert cost > 0  # 应该有估计成本

    # 测试未知模型
    connector.model = "unknown-model"
    cost = connector.estimate_cost("Test prompt", response)
    expected_cost = (1000 / 1000000) * 1.25 + (500 / 1000000) * 5.0  # 默认使用claude-3-haiku价格
    assert cost == pytest.approx(expected_cost)