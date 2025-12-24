"""
缓存管理模块
提供数据缓存功能，避免重复请求 akshare API
"""

from datetime import datetime
from typing import Optional, Any
import pandas as pd
import akshare as ak


class DataCache:
    """简单的数据缓存类，支持过期时间"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """获取缓存数据，超过max_age_seconds秒则返回None"""
        if key not in self._cache:
            return None
        
        cached_time = self._timestamps.get(key, datetime.min)
        if (datetime.now() - cached_time).total_seconds() > max_age_seconds:
            # 缓存过期，删除
            del self._cache[key]
            del self._timestamps[key]
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any):
        """设置缓存数据"""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._timestamps.clear()
    
    def stats(self) -> dict:
        """返回缓存统计信息"""
        return {
            'cache_size': len(self._cache),
            'keys': list(self._cache.keys())
        }


# 全局缓存实例
_cache = DataCache()

# 缓存过期时间配置（秒）
CACHE_TTL = {
    'etf_spot': 60,        # ETF实时行情缓存60秒
    'etf_hist': 300,       # ETF历史数据缓存5分钟
    'index_spot': 60,      # 指数实时行情缓存60秒
    'index_hist': 300,     # 指数历史数据缓存5分钟
    'macro': 3600,         # 宏观数据缓存1小时
    'calendar': 3600,      # 经济日历缓存1小时
}


def get_cached_etf_spot() -> pd.DataFrame:
    """获取ETF实时行情（带缓存）"""
    cache_key = 'etf_spot_em'
    cached = _cache.get(cache_key, CACHE_TTL['etf_spot'])
    if cached is not None:
        return cached
    
    df = ak.fund_etf_spot_em()
    _cache.set(cache_key, df)
    return df


def get_cached_index_spot_sina() -> pd.DataFrame:
    """获取指数实时行情-新浪（带缓存）"""
    cache_key = 'index_spot_sina'
    cached = _cache.get(cache_key, CACHE_TTL['index_spot'])
    if cached is not None:
        return cached
    
    df = ak.stock_zh_index_spot_sina()
    _cache.set(cache_key, df)
    return df


def get_cached_index_global_spot() -> pd.DataFrame:
    """获取全球指数实时行情（带缓存）"""
    cache_key = 'index_global_spot'
    cached = _cache.get(cache_key, CACHE_TTL['index_spot'])
    if cached is not None:
        return cached
    
    df = ak.index_global_spot_em()
    _cache.set(cache_key, df)
    return df


def get_cache() -> DataCache:
    """获取全局缓存实例"""
    return _cache


def clear_cache() -> dict:
    """清除缓存并返回统计"""
    stats = _cache.stats()
    _cache.clear()
    return stats


def get_cache_stats() -> dict:
    """获取缓存统计信息"""
    return _cache.stats()
