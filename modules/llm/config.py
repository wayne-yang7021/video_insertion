"""
LLM配置管理
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """LLM配置類"""
    
    # 基本配置
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash-exp"
    api_key: Optional[str] = None
    
    # 性能配置
    timeout_seconds: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        """初始化後處理"""
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")
    
    def is_enabled(self) -> bool:
        """檢查LLM功能是否可用"""
        return bool(self.api_key)


# 預設配置
DEFAULT_CONFIG = LLMConfig()


def get_llm_config() -> LLMConfig:
    """獲取LLM配置"""
    return LLMConfig()