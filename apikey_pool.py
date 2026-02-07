#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
from typing import List, Optional

class ApiKeyPool:

    def __init__(self, apikey_file: Optional[str] = None, keys: Optional[List[str]] = None):
        self.apikey_file = apikey_file
        self._lock = threading.Lock()
        if keys is not None:
            self._keys = self._normalize_keys(keys)
        elif apikey_file:
            self._keys = self._load_keys(apikey_file)
        else:
            self._keys = []
        self._idx: int = 0  

    @staticmethod
    def _load_keys(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            raw = [line.rstrip("\n") for line in f]
        return ApiKeyPool._normalize_keys(raw, source=f"API key file: {path}")

    @staticmethod
    def _normalize_keys(keys: List[str], source: str = "API key list") -> List[str]:
        # Accept newline/comma separated values and drop blanks/comments.
        cleaned: List[str] = []
        for item in keys:
            if not isinstance(item, str):
                continue
            for line in item.splitlines():
                parts = [seg.strip() for seg in line.split(",")]
                for seg in parts:
                    if not seg or seg.startswith("#"):
                        continue
                    cleaned.append(seg)

        seen = set()
        out: List[str] = []
        # De-duplicate while preserving input order for predictable rotation.
        for k in cleaned:
            if k not in seen:
                seen.add(k)
                out.append(k)

        if not out:
            raise RuntimeError(f"{source} is empty or contains comments only")

        return out

    def pick_key(self) -> str:
        # Thread-safe round-robin key selection.
        with self._lock:
            n = len(self._keys)
            if n == 0:
                raise RuntimeError("API key list is empty")
            key = self._keys[self._idx % n]
            self._idx = (self._idx + 1) % n
            return key

    def get_client(self):
        from google import genai
        key = self.pick_key()
        return genai.Client(api_key=key)

    def keys(self) -> List[str]:
        # Return a copy so callers cannot mutate internal state.
        with self._lock:
            return list(self._keys)

    def update_keys(self, keys: List[str]):
        with self._lock:
            self._keys = self._normalize_keys(keys)
            self._idx = 0


# Optional global pool for one-line client retrieval in small scripts.
_global_pool: Optional[ApiKeyPool] = None


def get_global_pool(apikey_file: Optional[str] = None, keys: Optional[List[str]] = None) -> ApiKeyPool:
    global _global_pool
    if _global_pool is None:
        _global_pool = ApiKeyPool(apikey_file=apikey_file, keys=keys)
    elif keys is not None:
        _global_pool.update_keys(keys)
    return _global_pool


def get_client(apikey_file: Optional[str] = None, keys: Optional[List[str]] = None):
    return get_global_pool(apikey_file=apikey_file, keys=keys).get_client()
