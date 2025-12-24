"""
数据获取模块
提供 ETF 和指数相关数据的获取函数
"""

from datetime import datetime, timedelta
import pandas as pd
import akshare as ak

from cache import (
    get_cached_etf_spot,
    get_cache,
    CACHE_TTL
)


def search_etf_by_name(name: str) -> list:
    """根据名称搜索ETF（使用缓存）"""
    try:
        # 使用缓存获取ETF列表
        etf_df = get_cached_etf_spot()
        
        # 模糊匹配名称
        matched = etf_df[etf_df['名称'].str.contains(name, case=False, na=False)]
        
        if matched.empty:
            return []
        
        result = []
        for _, row in matched.head(10).iterrows():
            result.append({
                'code': row['代码'],
                'name': row['名称'],
                'latest_price': row.get('最新价', 'N/A'),
                'change_pct': row.get('涨跌幅', 'N/A')
            })
        
        return result
    except Exception as e:
        return [{'error': str(e)}]


def get_etf_hist_data(code: str, days: int = 250) -> pd.DataFrame:
    """获取ETF历史数据（使用缓存）"""
    try:
        _cache = get_cache()
        
        # 生成缓存key（基于代码和天数）
        cache_key = f'etf_hist_{code}_{days}'
        cached = _cache.get(cache_key, CACHE_TTL['etf_hist'])
        if cached is not None:
            return cached
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名 (原始: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率)
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'change_pct', 'change_amount', 'turnover']
        df['date'] = pd.to_datetime(df['date'])
        
        # 确保数值列为float类型
        numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 存入缓存
        _cache.set(cache_key, df)
        
        return df
    except Exception as e:
        raise Exception(f"获取ETF历史数据失败: {str(e)}")


def get_index_hist_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """获取指数历史数据（使用缓存）"""
    _cache = get_cache()
    
    cache_key = f'index_hist_{symbol}_{days}'
    cached = _cache.get(cache_key, CACHE_TTL['index_hist'])
    if cached is not None:
        return cached
    
    df = ak.stock_zh_index_daily(symbol=symbol)
    if not df.empty:
        _cache.set(cache_key, df)
    
    return df
