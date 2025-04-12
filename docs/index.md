# BrainForge 文档

欢迎使用BrainForge文档！BrainForge是一个创新的LLM适配框架，通过"左脑"微调插件和"右脑"LLM API接口的协作，实现高效、低成本的大型语言模型定制能力。

## 什么是BrainForge？

BrainForge采用独特的双脑协作架构，将资源密集型的LLM推理与轻量级的适配层分离，使得即使没有完整模型访问权限，也能实现领域适配和性能优化。

**主要特点：**

- **参数高效微调**：利用LoRA、Prompt-Tuning等技术实现轻量级模型调整
- **多API支持**：无缝对接OpenAI、Anthropic、Cohere等API服务
- **领域适配**：专业术语处理、格式转换和内容增强
- **成本优化**：智能平衡性能与API调用成本

## 快速入门

### 安装

```bash
pip install brainforge
```
### 基本使用
```bash
pythonfrom brainforge import BrainForge
from brainforge.left_brain.adapters import LoRAAdapter
from brainforge.right_brain.connectors import OpenAIConnector
# 创建右脑API连接器
right_brain = OpenAIConnector(
    api_key="your_api_key",
    model="gpt-4"
)

# 创建左脑适配器
left_brain = LoRAAdapter(
    domain="medical",
    rank=8,
    alpha=16
)

# 初始化BrainForge
brain = BrainForge(right_brain=right_brain, left_brain=left_brain)

# 微调适配器
brain.train("path/to/medical_dataset.json", epochs=3)

# 使用微调后的系统
response = brain.generate("患者出现头痛、发热和咳嗽症状，可能的诊断是什么？")
print(response)
```
## 获取帮助
如果您有任何问题或需要帮助，可以：

- **查阅完整文档**
- **在GitHub Issues提交问题**
- **加入社区讨论**

## 贡献
我们欢迎各种形式的贡献！请查看贡献指南了解如何参与项目开发。