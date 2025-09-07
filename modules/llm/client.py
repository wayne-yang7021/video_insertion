"""
LLM客戶端封裝
"""

import time
import json
import logging
from typing import Optional, Dict, Any
from .config import LLMConfig, get_llm_config

# 設置日誌
logger = logging.getLogger(__name__)

# 導入Gemini客戶端
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not available")

class LLMClient:
    """統一的LLM客戶端"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化LLM客戶端
        
        Args:
            config: LLM配置，如果為None則使用預設配置
        """
        self.config = config or get_llm_config()
        self.client = None
        self.enabled = False
        
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化Gemini客戶端"""
        if not self.config.is_enabled():
            logger.warning("LLM API密鑰未設置，功能將停用")
            return
        
        try:
            if GEMINI_AVAILABLE:
                genai.configure(api_key=self.config.api_key)
                self.client = genai.GenerativeModel(self.config.model_name)
                self.enabled = True
                logger.info(f"Gemini客戶端初始化成功: {self.config.model_name}")
            else:
                logger.error("Gemini客戶端不可用")
                
        except Exception as e:
            logger.error(f"LLM客戶端初始化失敗: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """檢查LLM客戶端是否可用"""
        return self.enabled
    
    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        """
        生成文字回應
        
        Args:
            prompt: 輸入提示
            **kwargs: 額外參數
            
        Returns:
            Optional[str]: 生成的文字，失敗時返回None
        """
        if not self.enabled:
            logger.warning("LLM客戶端未啟用")
            return None
        
        try:
            return self._call_with_retry(prompt, **kwargs)
        except Exception as e:
            logger.error(f"LLM文字生成失敗: {e}")
            return None
    
    def _call_with_retry(self, prompt: str, **kwargs) -> Optional[str]:
        """帶重試機制的Gemini調用"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.generate_content(prompt)
                return response.text if response else None
                    
            except Exception as e:
                error_msg = str(e)
                
                # 檢查是否是額度限制錯誤
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < self.config.max_retries - 1:
                        wait_time = 2 ** attempt * 5  # 指數退避
                        logger.warning(f"API額度限制，等待{wait_time}秒後重試...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("API額度已用完")
                        return None
                else:
                    # 其他錯誤
                    if attempt < self.config.max_retries - 1:
                        logger.warning(f"LLM調用失敗: {e}，重試中...")
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"LLM調用最終失敗: {e}")
                        return None
        
        return None
    
    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        解析JSON格式的回應
        
        Args:
            response: LLM的文字回應
            
        Returns:
            Optional[Dict]: 解析後的JSON對象，失敗時返回None
        """
        if not response:
            return None
        
        try:
            # 清理回應文字，移除markdown代碼塊標記
            clean_text = response.strip()
            
            # 移除```json和```標記
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.startswith('```'):
                clean_text = clean_text[3:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            
            clean_text = clean_text.strip()
            
            return json.loads(clean_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失敗: {e}")
            
            # 嘗試使用正則表達式提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            return None


# 便利函數
def create_llm_client() -> LLMClient:
    """創建LLM客戶端實例"""
    config = get_llm_config()
    return LLMClient(config)


def is_llm_available() -> bool:
    """檢查LLM功能是否可用"""
    client = create_llm_client()
    return client.is_enabled()