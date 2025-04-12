# BrainForge: 左右脑协作的高效LLM微调系统


[![GitHub license](https://img.shields.io/github/license/yourusername/brainforge.svg)](https://github.com/yourusername/brainforge/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**BrainForge** 是一个创新的LLM适配框架，通过"左脑"微调插件和"右脑"LLM API接口的协作，实现高效、低成本的大型语言模型定制能力。无需直接访问完整模型参数，也能实现领域适配和性能优化。

## 特点

- 🧠 **参数高效微调**：LoRA、Prompt-Tuning等技术实现轻量级模型调整
- 🔌 **多API支持**：无缝对接OpenAI、Anthropic、Cohere等API服务
- 📊 **领域适配**：专业术语处理、格式转换和内容增强
- 💰 **成本优化**：智能平衡性能与API调用成本

## 快速开始

```bash
# 安装
pip install brainforge
````
```bash
# 基本使用
from brainforge import BrainForge
from brainforge.plugins import LoRAAdapter
from brainforge.connectors import OpenAIConnector
```
```bash
# 初始化
brain = BrainForge(
    right_brain=OpenAIConnector(api_key="your_key", model="gpt-4"),
    left_brain=LoRAAdapter(domain="medical")
)
```
```bash
# 微调
brain.train("path/to/data.json")
```
```bash
# 生成
response = brain.generate("您好，我想咨询一下...")
```
## 文档
详细文档请参见BrainForge文档
## 许可证
本项目采用MIT许可证
