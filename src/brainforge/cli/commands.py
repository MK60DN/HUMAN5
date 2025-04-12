"""
BrainForge命令行界面。
"""

import os
import sys
import json
import logging
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.logging import RichHandler

from ..core.brain import BrainForge
from ..core.config import BrainForgeConfig
from ..left_brain.adapters.lora import LoRAAdapter
from ..left_brain.adapters.prompt_tuning import PromptTuningAdapter
from ..right_brain.connectors.openai import OpenAIConnector
from ..right_brain.connectors.anthropic import AnthropicConnector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("brainforge.cli")
console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """BrainForge: 左右脑协作的高效LLM微调系统"""
    pass


@cli.command()
@click.argument("project_name")
def init(project_name: str):
    """初始化一个新项目"""
    try:
        # 创建项目目录
        if os.path.exists(project_name):
            console.print(f"[yellow]警告: 目录 '{project_name}' 已存在[/yellow]")
            if not click.confirm("是否继续?"):
                return
        else:
            os.makedirs(project_name)

        # 创建子目录
        os.makedirs(os.path.join(project_name, "data"), exist_ok=True)
        os.makedirs(os.path.join(project_name, "models"), exist_ok=True)
        os.makedirs(os.path.join(project_name, "logs"), exist_ok=True)

        # 创建配置文件
        config = {
            "project_name": project_name,
            "brainforge_version": "0.1.0",
            "api": {
                "provider": None,
                "key": None,
                "model": None
            },
            "adapter": {
                "type": None,
                "domain": "general",
                "parameters": {}
            },
            "training": {
                "data_path": "data/train.json",
                "epochs": 3,
                "batch_size": 8
            }
        }

        with open(os.path.join(project_name, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 创建示例数据文件
        example_data = [
            {"input": "示例输入1", "output": "示例输出1"},
            {"input": "示例输入2", "output": "示例输出2"}
        ]

        with open(os.path.join(project_name, "data", "example.json"), "w") as f:
            json.dump(example_data, f, indent=2)

        console.print(f"[green]已成功初始化项目: {project_name}[/green]")
        console.print(f"下一步推荐执行: [bold]brainforge config api --project {project_name}[/bold]")

    except Exception as e:
        console.print(f"[red]初始化项目失败: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("component")
@click.option("--project", "-p", help="项目目录")
@click.option("--provider", help="API提供商(openai/anthropic/cohere)")
@click.option("--key", help="API密钥")
@click.option("--model", help="模型名称")
@click.option("--adapter", help="适配器类型(lora/prompt_tuning/qlora)")
@click.option("--domain", help="适配领域")
@click.option("--param", "-P", multiple=True, help="其他参数(格式: name=value)")
def config(
        component: str,
        project: Optional[str] = None,
        provider: Optional[str] = None,
        key: Optional[str] = None,
        model: Optional[str] = None,
        adapter: Optional[str] = None,
        domain: Optional[str] = None,
        param: List[str] = None
):
    """配置项目组件(api/adapter)"""
    try:
        # 确定项目目录
        if project is None:
            project = os.getcwd()

        # 加载配置
        config_path = os.path.join(project, "config.json")
        if not os.path.exists(config_path):
            console.print(f"[red]错误: 找不到配置文件 {config_path}[/red]")
            console.print(f"提示: 使用 [bold]brainforge init[/bold] 创建新项目，或指定正确的项目路径")
            sys.exit(1)

        with open(config_path, "r") as f:
            config_data = json.load(f)

        # 处理参数
        params = {}
        if param:
            for p in param:
                if "=" in p:
                    k, v = p.split("=", 1)
                    # 尝试转换为数值
                    try:
                        if "." in v:
                            params[k] = float(v)
                        else:
                            params[k] = int(v)
                    except ValueError:
                        params[k] = v

        # 根据组件类型更新配置
        if component.lower() == "api":
            if provider:
                config_data["api"]["provider"] = provider
            if key:
                config_data["api"]["key"] = key
            if model:
                config_data["api"]["model"] = model
            for k, v in params.items():
                config_data["api"][k] = v

            console.print("[green]API配置已更新[/green]")

        elif component.lower() == "adapter":
            if adapter:
                config_data["adapter"]["type"] = adapter
            if domain:
                config_data["adapter"]["domain"] = domain
            for k, v in params.items():
                config_data["adapter"]["parameters"][k] = v

            console.print("[green]适配器配置已更新[/green]")

        else:
            console.print(f"[red]错误: 未知组件类型 '{component}'[/red]")
            console.print("支持的组件: api, adapter")
            sys.exit(1)

        # 保存配置
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    except Exception as e:
        console.print(f"[red]配置失败: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--project", "-p", help="项目目录")
@click.option("--adapter", help="适配器类型(lora/prompt_tuning/qlora)")
@click.option("--domain", help="适配领域")
@click.option("--data", "-d", help="训练数据路径")
@click.option("--epochs", "-e", type=int, help="训练轮数")
@click.option("--batch-size", "-b", type=int, help="批处理大小")
@click.option("--param", "-P", multiple=True, help="其他参数(格式: name=value)")
def train(
        project: Optional[str] = None,
        adapter: Optional[str] = None,
        domain: Optional[str] = None,
        data: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        param: List[str] = None
):
    """训练左脑适配器"""
    try:
        import asyncio

        # 确定项目目录
        if project is None:
            project = os.getcwd()

        # 加载配置
        config_path = os.path.join(project, "config.json")
        if not os.path.exists(config_path):
            console.print(f"[red]错误: 找不到配置文件 {config_path}[/red]")
            sys.exit(1)

        with open(config_path, "r") as f:
            config_data = json.load(f)

        # 使用参数覆盖配置
        if adapter:
            config_data["adapter"]["type"] = adapter
        if domain:
            config_data["adapter"]["domain"] = domain
        if data:
            config_data["training"]["data_path"] = data
        if epochs:
            config_data["training"]["epochs"] = epochs
        if batch_size:
            config_data["training"]["batch_size"] = batch_size

        # 处理附加参数
        adapter_params = config_data["adapter"]["parameters"].copy()
        if param:
            for p in param:
                if "=" in p:
                    k, v = p.split("=", 1)
                    # 尝试转换为数值
                    try:
                        if "." in v:
                            adapter_params[k] = float(v)
                        else:
                            adapter_params[k] = int(v)
                    except ValueError:
                        adapter_params[k] = v

        # 检查必要配置
        if not config_data["adapter"]["type"]:
            console.print("[red]错误: 未指定适配器类型[/red]")
            sys.exit(1)

        if not config_data["api"]["provider"] or not config_data["api"]["key"]:
            console.print("[red]错误: API未完全配置[/red]")
            console.print("提示: 使用 [bold]brainforge config api[/bold] 配置API")
            sys.exit(1)

        # 确定数据路径
        data_path = config_data["training"]["data_path"]
        if not os.path.isabs(data_path):
            data_path = os.path.join(project, data_path)

        if not os.path.exists(data_path):
            console.print(f"[red]错误: 找不到训练数据 {data_path}[/red]")
            sys.exit(1)

        # 创建适配器
        adapter_type = config_data["adapter"]["type"].lower()
        adapter_domain = config_data["adapter"]["domain"]

        if adapter_type == "lora":
            left_brain = LoRAAdapter(domain=adapter_domain, **adapter_params)
        elif adapter_type == "prompt_tuning":
            left_brain = PromptTuningAdapter(domain=adapter_domain, **adapter_params)
        else:
            console.print(f"[red]错误: 不支持的适配器类型 '{adapter_type}'[/red]")
            sys.exit(1)

        # 创建API连接器
        api_provider = config_data["api"]["provider"].lower()
        api_key = config_data["api"]["key"]
        api_model = config_data["api"]["model"]

        if api_provider == "openai":
            right_brain = OpenAIConnector(api_key=api_key, model=api_model)
        elif api_provider == "anthropic":
            right_brain = AnthropicConnector(api_key=api_key, model=api_model)
        else:
            console.print(f"[red]错误: 不支持的API提供商 '{api_provider}'[/red]")
            sys.exit(1)

        # 创建BrainForge实例
        brain = BrainForge(right_brain=right_brain, left_brain=left_brain)

        # 开始训练
        console.print(f"[bold green]开始训练适配器...[/bold green]")
        console.print(f"- 适配器类型: {adapter_type}")
        console.print(f"- 适配领域: {adapter_domain}")
        console.print(f"- 数据路径: {data_path}")
        console.print(f"- 训练轮数: {config_data['training']['epochs']}")
        console.print(f"- 批处理大小: {config_data['training']['batch_size']}")

        with console.status("[bold green]训练中...[/bold green]"):
            results = brain.train(
                data_path=data_path,
                epochs=config_data["training"]["epochs"],
                batch_size=config_data["training"]["batch_size"]
            )

        # 显示结果
        console.print("[bold green]训练完成![/bold green]")
        console.print("训练结果:")
        for k, v in results.items():
            console.print(f"- {k}: {v}")

        # 保存模型
        model_dir = os.path.join(project, "models", f"{adapter_type}_{adapter_domain}")
        os.makedirs(model_dir, exist_ok=True)

        console.print(f"[bold]保存模型到 {model_dir}[/bold]")
        brain.save(model_dir)

        console.print("[green]模型保存成功[/green]")
        console.print(f"下一步推荐执行: [bold]brainforge serve --model {model_dir}[/bold]")

    except Exception as e:
        console.print(f"[red]训练失败: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", required=True, help="模型路径")
@click.option("--api-key", help="API密钥")
@click.option("--api-model", help="API模型名称")
@click.option("--port", "-p", type=int, default=8000, help="服务端口")
@click.option("--host", default="127.0.0.1", help="服务主机")
def serve(
    model: str,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None,
    port: int = 8000,
    host: str = "127.0.0.1"
):
    """启动推理服务"""
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        # 检查模型路径
        if not os.path.exists(model):
            console.print(f"[red]错误: 找不到模型目录 {model}[/red]")
            sys.exit(1)

        # 加载模型信息
        info_path = os.path.join(model, "left_brain", "info.json")
        with open(info_path, "r") as f:
            adapter_info = json.load(f)

        adapter_type = adapter_info["type"]
        adapter_domain = adapter_info["domain"]

        # 加载配置
        config_path = os.path.join(model, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # 创建API连接器
        # 如果提供了API密钥和模型，则使用这些值覆盖保存的配置
        if api_key is not None:
            config["api_key"] = api_key
        if api_model is not None:
            config["model"] = api_model

        # 确保有API密钥
        if "api_key" not in config:
            console.print("[red]错误: 未提供API密钥[/red]")
            console.print("提示: 使用 --api-key 参数指定API密钥")
            sys.exit(1)

        # 根据适配器类型确定右脑连接器
        # 这里假设我们能从配置中确定使用哪个API
        if "openai" in config.get("model", "").lower() or config.get("provider", "").lower() == "openai":
            right_brain = OpenAIConnector(
                api_key=config["api_key"],
                model=config.get("model", "gpt-4")
            )
        else:
            right_brain = AnthropicConnector(
                api_key=config["api_key"],
                model=config.get("model", "claude-3-opus-20240229")
            )

        # 加载BrainForge模型
        console.print(f"[bold]加载模型: {adapter_type}_{adapter_domain}[/bold]")
        brain = BrainForge.load(model, right_brain=right_brain)

        # 创建FastAPI应用
        app = FastAPI(
            title="BrainForge API",
            description=f"BrainForge 左右脑协作LLM微调系统 API - {adapter_type}/{adapter_domain}",
            version="0.1.0"
        )

        # 定义请求模型
        class GenerateRequest(BaseModel):
            prompt: str
            temperature: float = 0.7
            max_tokens: Optional[int] = None

        class GenerateResponse(BaseModel):
            text: str
            metadata: Dict[str, Any]

        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            try:
                result = brain.generate(
                    prompt=request.prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )

                # 包装结果
                return {
                    "text": result,
                    "metadata": {
                        "model": f"{adapter_type}/{adapter_domain}",
                        "api_model": right_brain.model
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # 启动服务
        console.print(f"[bold green]启动BrainForge服务[/bold green]")
        console.print(f"- 模型: {adapter_type}/{adapter_domain}")
        console.print(f"- API模型: {right_brain.model}")
        console.print(f"- 地址: http://{host}:{port}")
        console.print("[yellow]按Ctrl+C停止服务[/yellow]")

        uvicorn.run(app, host=host, port=port)

    except Exception as e:
        console.print(f"[red]启动服务失败: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option("--test", "-t", required=True, help="测试数据路径")
@click.option("--model", "-m", required=True, help="模型路径")
@click.option("--api-key", help="API密钥")
@click.option("--api-model", help="API模型名称")
@click.option("--metrics", "-M", default="accuracy,relevance", help="评估指标(逗号分隔)")
def evaluate(
    test: str,
    model: str,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None,
    metrics: str = "accuracy,relevance"
):
    """评估模型性能"""
    try:
        # 检查测试数据路径
        if not os.path.exists(test):
            console.print(f"[red]错误: 找不到测试数据 {test}[/red]")
            sys.exit(1)

        # 检查模型路径
        if not os.path.exists(model):
            console.print(f"[red]错误: 找不到模型目录 {model}[/red]")
            sys.exit(1)

        # 加载模型信息
        info_path = os.path.join(model, "left_brain", "info.json")
        with open(info_path, "r") as f:
            adapter_info = json.load(f)

        adapter_type = adapter_info["type"]
        adapter_domain = adapter_info["domain"]

        # 加载配置
        config_path = os.path.join(model, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # 创建API连接器
        # 如果提供了API密钥和模型，则使用这些值覆盖保存的配置
        if api_key is not None:
            config["api_key"] = api_key
        if api_model is not None:
            config["model"] = api_model

        # 确保有API密钥
        if "api_key" not in config:
            console.print("[red]错误: 未提供API密钥[/red]")
            console.print("提示: 使用 --api-key 参数指定API密钥")
            sys.exit(1)

        # 根据适配器类型确定右脑连接器
        # 这里假设我们能从配置中确定使用哪个API
        if "openai" in config.get("model", "").lower() or config.get("provider", "").lower() == "openai":
            right_brain = OpenAIConnector(
                api_key=config["api_key"],
                model=config.get("model", "gpt-4")
            )
        else:
            right_brain = AnthropicConnector(
                api_key=config["api_key"],
                model=config.get("model", "claude-3-opus-20240229")
            )

        # 加载BrainForge模型
        console.print(f"[bold]加载模型: {adapter_type}_{adapter_domain}[/bold]")
        brain = BrainForge.load(model, right_brain=right_brain)

        # 解析指标
        metrics_list = [m.strip() for m in metrics.split(",")]

        # 开始评估
        console.print(f"[bold green]开始评估...[/bold green]")
        console.print(f"- 模型: {adapter_type}/{adapter_domain}")
        console.print(f"- API模型: {right_brain.model}")
        console.print(f"- 测试数据: {test}")
        console.print(f"- 评估指标: {metrics}")

        with console.status("[bold green]评估中...[/bold green]"):
            results = brain.evaluate(test, metrics=metrics_list)

        # 显示结果
        console.print("[bold green]评估完成![/bold green]")
        console.print("评估结果:")
        for k, v in results.items():
            console.print(f"- {k}: {v:.4f}")

    except Exception as e:
        console.print(f"[red]评估失败: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    cli()