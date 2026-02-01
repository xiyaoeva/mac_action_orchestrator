#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
from typing import List, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ApiKeyPool:
    """
    极简轮转 API Key 池（round-robin，无冷冻/ban）：
    - 每次 pick_key() 返回下一个 key
    - 永远不会连续两次选同一个 key（除非只有 1 个 key）
    - 线程安全
    """

    def __init__(self, apikey_file: str = os.path.join(BASE_DIR, "apikeys.txt")):
        self.apikey_file = apikey_file
        self._lock = threading.Lock()
        self._keys: List[str] = self._load_keys(apikey_file)
        self._idx: int = 0  # 下一个要返回的 key 下标

    @staticmethod
    def _load_keys(path: str) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"API key 文件不存在: {path}")

        keys: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                k = line.strip()
                if not k or k.startswith("#"):
                    continue
                keys.append(k)

        # 去重但保持顺序
        seen = set()
        out: List[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)

        if not out:
            raise RuntimeError(f"API key 文件为空或全是注释: {path}")

        return out

    def pick_key(self) -> str:
        with self._lock:
            n = len(self._keys)
            if n == 0:
                raise RuntimeError("API key 列表为空")
            key = self._keys[self._idx % n]
            self._idx = (self._idx + 1) % n
            return key

    def get_client(self):
        from google import genai
        key = self.pick_key()
        return genai.Client(api_key=key)

    def keys(self) -> List[str]:
        # 返回 key 列表（只读副本）
        with self._lock:
            return list(self._keys)


# ===== 可选：全局池（方便一行取 client）=====
_global_pool: Optional[ApiKeyPool] = None


def get_global_pool(apikey_file: str = os.path.join(BASE_DIR, "apikeys.txt")) -> ApiKeyPool:
    global _global_pool
    if _global_pool is None:
        _global_pool = ApiKeyPool(apikey_file=apikey_file)
    return _global_pool


def get_client(apikey_file: str = os.path.join(BASE_DIR, "apikeys.txt")):
    return get_global_pool(apikey_file=apikey_file).get_client()
