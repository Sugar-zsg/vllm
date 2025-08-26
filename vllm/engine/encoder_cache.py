# encoder_cache.py

import torch
from threading import Lock

class EncoderCache:
    _instance = None
    _lock = Lock()  # 线程安全

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._encoder_output = None
        return cls._instance

    def set_encoder_output(self, encoder_output: torch.Tensor):
        """保存 encoder 输出"""
        if not isinstance(encoder_output, torch.Tensor):
            raise TypeError("encoder_output must be a torch.Tensor")
        self._encoder_output = encoder_output

    def get_encoder_output(self) -> torch.Tensor | None:
        """获取缓存的 encoder 输出，如果没有设置则返回 None"""
        return self._encoder_output

    def release_encoder_output(self):
        """释放 encoder 输出，清理显存"""
        self._encoder_output = None
        torch.cuda.empty_cache()
