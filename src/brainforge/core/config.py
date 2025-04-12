"""
BrainForge配置管理。
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class BrainForgeConfig(BaseModel):
    """BrainForge配置类"""

    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: Optional[str] = Field(default=None, description="日志文件路径")

    # 缓存配置
    cache_enabled: bool = Field(default=True, description="是否启用缓存")
    max_cache_size: int = Field(default=1000, description="最大缓存条目数")
    cache_ttl: int = Field(default=3600, description="缓存生存时间(秒)")

    # 性能跟踪
    track_metrics: bool = Field(default=True, description="是否跟踪性能指标")

    # 并发配置
    max_concurrent_requests: int = Field(default=10, description="最大并发请求数")

    # 重试配置
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟(秒)")

    # 微调配置
    default_train_epochs: int = Field(default=3, description="默认训练轮数")
    default_batch_size: int = Field(default=8, description="默认批处理大小")

    # 路径配置
    model_save_path: str = Field(default="./models", description="模型保存路径")
    data_path: str = Field(default="./data", description="数据路径")

    # 其他配置
    device: str = Field(default="auto", description="计算设备(auto, cpu, cuda)")

    def dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return super().model_dump()


# 默认配置
DEFAULT_CONFIG = BrainForgeConfig()