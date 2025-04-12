"""
适配器单元测试。
"""

import pytest
import os
import json
import shutil
from unittest.mock import AsyncMock, patch

from brainforge.left_brain.adapters.base import BaseAdapter
from brainforge.left_brain.adapters.lora import LoRAAdapter
from brainforge.left_brain.adapters.prompt_tuning import PromptTuningAdapter


# 设置测试数据
@pytest.fixture
def sample_data():
    return [
        {"input": "Hello", "output": "Hi there!"},
        {"input": "How are you?", "output": "I'm doing well, thank you!"}
    ]


@pytest.fixture
def temp_dir(tmpdir):
    """创建临时目录"""
    yield tmpdir
    # 清理
    shutil.rmtree(tmpdir)


# 测试LoRA适配器
@pytest.mark.asyncio
async def test_lora_adapter_basic():
    """测试LoRA适配器基本功能"""
    # 使用小模型进行测试，避免大模型加载
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # 模拟模型加载
            mock_model.return_value = AsyncMock()
            mock_tokenizer.return_value = AsyncMock()

            # 创建适配器
            adapter = LoRAAdapter(
                domain="test",
                rank=4,
                alpha=16,
                base_model_name="facebook/opt-125m"
            )

            # 测试初始化
            assert adapter.domain == "test"
            assert adapter.rank == 4
            assert adapter.alpha == 16
            assert adapter.is_trained is False

            # 测试处理输入
            with patch.object(adapter, 'is_trained', True):
                result = await adapter.process_input("Test input")
                assert isinstance(result, str)
                assert "Test input" in result


@pytest.mark.asyncio
async def test_lora_adapter_train(sample_data):
    """测试LoRA适配器训练功能"""
    # 使用小模型进行测试，避免大模型加载
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            with patch('transformers.Trainer') as mock_trainer:
                # 模拟模型加载和训练
                mock_model.return_value = AsyncMock()
                mock_tokenizer.return_value = AsyncMock()

                # 模拟训练结果
                mock_train_result = type('obj', (object,), {
                    'training_loss': 0.5,
                    'metrics': {
                        'train_runtime': 10.0,
                        'train_samples_per_second': 5.0
                    }
                })
                mock_trainer_instance = AsyncMock()
                mock_trainer_instance.train.return_value = mock_train_result
                mock_trainer.return_value = mock_trainer_instance

                # 创建适配器
                adapter = LoRAAdapter(
                    domain="test",
                    rank=4,
                    alpha=16,
                    base_model_name="facebook/opt-125m"
                )

                # 测试训练方法
                result = await adapter.train_async(sample_data, epochs=1, batch_size=2)

                # 验证结果
                assert "train_loss" in result
                assert isinstance(result["train_loss"], float)
                assert "train_runtime" in result
                assert result["epochs"] == 1
                assert result["trained_samples"] == len(sample_data)
                assert adapter.is_trained is True


@pytest.mark.asyncio
async def test_lora_adapter_save_load(temp_dir):
    """测试LoRA适配器保存和加载功能"""
    # 使用小模型进行测试，避免大模型加载
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # 模拟模型加载
            mock_model.return_value = AsyncMock()
            mock_tokenizer.return_value = AsyncMock()

            # 创建适配器
            adapter = LoRAAdapter(
                domain="test",
                rank=4,
                alpha=16,
                base_model_name="facebook/opt-125m"
            )

            # 设置训练标志
            adapter.is_trained = True

            # 保存适配器
            save_path = os.path.join(temp_dir, "lora_adapter")
            adapter.save(save_path)

            # 验证保存的文件
            assert os.path.exists(os.path.join(save_path, "info.json"))
            assert os.path.exists(os.path.join(save_path, "config.json"))

            # 加载适配器
            with patch.object(LoRAAdapter, '_initialize_model'):
                loaded_adapter = LoRAAdapter.load(save_path)

                # 验证加载的适配器
                assert loaded_adapter.domain == "test"
                assert loaded_adapter.rank == 4
                assert loaded_adapter.alpha == 16
                assert loaded_adapter.base_model_name == "facebook/opt-125m"


# 测试PromptTuning适配器
@pytest.mark.asyncio
async def test_prompt_tuning_adapter_basic():
    """测试PromptTuning适配器基本功能"""
    # 使用小模型进行测试，避免大模型加载
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # 模拟模型加载
            mock_model.return_value = AsyncMock()
            mock_tokenizer.return_value = AsyncMock()

            # 创建适配器
            adapter = PromptTuningAdapter(
                domain="test",
                num_virtual_tokens=10,
                prompt_tuning_init="TEXT",
                base_model_name="facebook/opt-125m"
            )

            # 测试初始化
            assert adapter.domain == "test"
            assert adapter.num_virtual_tokens == 10
            assert adapter.prompt_tuning_init == "TEXT"
            assert adapter.is_trained is False

            # 测试处理输入
            result = await adapter.process_input("Test input")
            assert result == "Test input"  # Prompt-Tuning不修改输入


@pytest.mark.asyncio
async def test_prompt_tuning_adapter_output_processing():
    """测试PromptTuning适配器输出处理功能"""
    # 使用小模型进行测试，避免大模型加载
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # 模拟模型加载
            mock_model.return_value = AsyncMock()
            mock_tokenizer.return_value = AsyncMock()

            # 创建适配器
            adapter = PromptTuningAdapter(
                domain="test",
                num_virtual_tokens=10,
                prompt_tuning_init="TEXT",
                base_model_name="facebook/opt-125m"
            )

            # 测试不同输出格式的处理
            # 字典格式API输出
            dict_output = {"text": "API result"}
            result = await adapter.process_output(dict_output, "Original input")
            assert result == "API result"

            # OpenAI格式API输出
            openai_output = {
                "choices": [{"message": {"content": "OpenAI result"}}]
            }
            result = await adapter.process_output(openai_output, "Original input")
            assert result == "OpenAI result"

            # 字符串格式API输出
            result = await adapter.process_output("Plain text result", "Original input")
            assert result == "Plain text result"


@pytest.mark.asyncio
async def test_prompt_tuning_adapter_error_handling():
    """测试PromptTuning适配器错误处理功能"""
    # 使用小模型进行测试，避免大模型加载
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # 模拟模型加载
            mock_model.return_value = AsyncMock()
            mock_tokenizer.return_value = AsyncMock()

            # 创建适配器
            adapter = PromptTuningAdapter(
                domain="test",
                num_virtual_tokens=10,
                prompt_tuning_init="TEXT",
                base_model_name="facebook/opt-125m"
            )

            # 测试输出处理错误处理
            # 无效字典
            result = await adapter.process_output({"invalid": "structure"}, "Original input")
            assert result == "{'invalid': 'structure'}"

            # 异常情况
            with patch.object(adapter, 'logger'):
                def raise_error(*args, **kwargs):
                    raise ValueError("Test error")

                # 替换方法为抛出异常的版本
                original_method = adapter.process_output
                adapter.process_output = raise_error

                # 恢复方法
                adapter.process_output = original_method