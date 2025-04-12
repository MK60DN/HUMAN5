"""
集成测试。
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import AsyncMock, patch, MagicMock

from brainforge import BrainForge
from brainforge.left_brain.adapters.lora import LoRAAdapter
from brainforge.right_brain.connectors.openai import OpenAIConnector


# 创建临时目录用于测试
@pytest.fixture
def temp_dir():
    """创建临时目录"""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    # 清理
    shutil.rmtree(tmp_dir)


# 测试BrainForge基本功能
@pytest.mark.asyncio
async def test_brainforge_init():
    """测试BrainForge初始化"""
    # 使用模拟对象
    with patch('transformers.AutoModelForCausalLM.from_pretrained'):
        with patch('transformers.AutoTokenizer.from_pretrained'):
            # 创建左脑和右脑组件
            left_brain = LoRAAdapter(domain="test")
            right_brain = MagicMock(spec=OpenAIConnector)

            # 创建BrainForge实例
            brain = BrainForge(right_brain=right_brain, left_brain=left_brain)

            # 验证初始化
            assert brain.left_brain == left_brain
            assert brain.right_brain == right_brain
            assert brain.bridge is not None


@pytest.mark.asyncio
async def test_brainforge_generate():
    """测试BrainForge生成功能"""
    # 使用模拟对象
    with patch('transformers.AutoModelForCausalLM.from_pretrained'):
        with patch('transformers.AutoTokenizer.from_pretrained'):
            # 创建左脑和右脑组件
            left_brain = LoRAAdapter(domain="test")
            right_brain = MagicMock(spec=OpenAIConnector)

            # 设置模拟方法返回值
            left_brain.process_input = AsyncMock(return_value="Processed input")
            left_brain.process_output = AsyncMock(return_value="Final output")

            right_brain.generate = AsyncMock(return_value={"text": "API response"})

            # 创建BrainForge实例
            brain = BrainForge(right_brain=right_brain, left_brain=left_brain)

            # 设置桥接层方法返回值
            brain.bridge.process = AsyncMock(return_value={
                "output": "Processed output"
            })

            # 测试生成方法
            result = await brain._generate_async("Test prompt")

            # 验证结果
            assert result == "Processed output"
            assert brain.bridge.process.called


@pytest.mark.asyncio
async def test_brainforge_train():
    """测试BrainForge训练功能"""
    # 使用模拟对象
    with patch('transformers.AutoModelForCausalLM.from_pretrained'):
        with patch('transformers.AutoTokenizer.from_pretrained'):
            with patch('brainforge.data.loader.load_data') as mock_load_data:
                # 设置模拟数据加载返回值
                test_data = [
                    {"input": "Test input 1", "output": "Test output 1"},
                    {"input": "Test input 2", "output": "Test output 2"}
                ]
                mock_load_data.return_value = test_data

                # 创建左脑和右脑组件
                left_brain = LoRAAdapter(domain="test")
                right_brain = MagicMock(spec=OpenAIConnector)

                # 设置模拟训练方法返回值
                train_result = {
                    "train_loss": 0.5,
                    "train_runtime": 10.0,
                    "epochs": 3,
                    "trained_samples": 2
                }
                left_brain.train_async = AsyncMock(return_value=train_result)

                # 创建BrainForge实例
                brain = BrainForge(right_brain=right_brain, left_brain=left_brain)

                # 测试训练方法
                result = await brain._train_async("dummy_path.json", epochs=3, batch_size=2)

                # 验证结果
                assert result == train_result
                assert left_brain.train_async.called
                left_brain.train_async.assert_called_with(
                    test_data, epochs=3, batch_size=2
                )


@pytest.mark.asyncio
async def test_brainforge_save_load(temp_dir):
    """测试BrainForge保存和加载功能"""
    # 使用模拟对象
    with patch('transformers.AutoModelForCausalLM.from_pretrained'):
        with patch('transformers.AutoTokenizer.from_pretrained'):
            with patch.object(LoRAAdapter, 'save'):
                with patch.object(LoRAAdapter, 'load') as mock_load:
                    # 创建左脑和右脑组件
                    left_brain = LoRAAdapter(domain="test")
                    right_brain = MagicMock(spec=OpenAIConnector)

                    # 设置加载方法返回值
                    mock_load.return_value = left_brain

                    # 创建BrainForge实例
                    brain = BrainForge(right_brain=right_brain, left_brain=left_brain)

                    # 保存模型
                    save_path = os.path.join(temp_dir, "test_model")
                    brain.save(save_path)

                    # 确保存储目录被创建
                    assert os.path.exists(save_path)

                    # 模拟adapter_info.json文件
                    os.makedirs(os.path.join(save_path, "left_brain"), exist_ok=True)
                    with open(os.path.join(save_path, "left_brain", "info.json"), "w") as f:
                        json.dump({"type": "LoRA", "domain": "test"}, f)

                    # 模拟config.json文件
                    with open(os.path.join(save_path, "config.json"), "w") as f:
                        json.dump({}, f)

                    # 测试加载方法
                    with patch('importlib.import_module') as mock_import:
                        # 模拟动态导入
                        mock_module = MagicMock()
                        mock_module.LoRAAdapter = LoRAAdapter
                        mock_import.return_value = mock_module

                        # 加载模型
                        loaded_brain = BrainForge.load(save_path, right_brain=right_brain)

                        # 验证结果
                        assert isinstance(loaded_brain, BrainForge)
                        assert mock_load.called