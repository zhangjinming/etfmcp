"""
数据获取模块
提供 ETF 和指数相关数据的获取函数
支持从 CSV 文件加载数据，如果没有则从 API 下载
"""

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import akshare as ak

from cache import (
    get_cached_etf_spot,
    get_cache,
    CACHE_TTL
)

# CSV 数据存储目录
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _get_csv_path(data_type: str, code: str) -> Path:
    """获取 CSV 文件路径"""
    return DATA_DIR / f"{data_type}_{code}.csv"


def _load_from_csv(csv_path: Path) -> pd.DataFrame:
    """从 CSV 文件加载数据"""
    if not csv_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame()


def _save_to_csv(df: pd.DataFrame, csv_path: Path):
    """保存数据到 CSV 文件"""
    if df.empty:
        return
    
    try:
        # 如果有 date 列，转换为字符串格式保存
        df_save = df.copy()
        if 'date' in df_save.columns:
            df_save['date'] = df_save['date'].dt.strftime('%Y-%m-%d')
        df_save.to_csv(csv_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"保存 CSV 文件失败: {e}")


def _merge_and_update_csv(existing_df: pd.DataFrame, new_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """合并现有数据和新数据，并更新 CSV 文件"""
    if existing_df.empty:
        _save_to_csv(new_df, csv_path)
        return new_df
    
    if new_df.empty:
        return existing_df
    
    # 合并数据，去重
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date'], keep='last')
    combined = combined.sort_values('date').reset_index(drop=True)
    
    # 保存更新后的数据
    _save_to_csv(combined, csv_path)
    
    return combined


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


def get_etf_hist_data(code: str, days: int = 250, start_date: str = "", end_date: str = "") -> pd.DataFrame:
    """
    获取ETF历史数据（优先从CSV加载，不足则从API下载）
    
    数据加载策略：
    1. 优先从本地 CSV 文件加载数据
    2. 如果 CSV 数据不足（时间范围不够），则从网络增量下载
    3. 下载的数据会合并到 CSV 文件中，供下次使用
    
    Args:
        code: ETF代码
        days: 获取最近多少天的数据（当start_date和end_date为空时使用）
        start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD
        end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD
    
    Returns:
        包含历史数据的DataFrame
    """
    try:
        _cache = get_cache()
        
        # 标准化日期格式（去除横杠）
        if start_date:
            start_date = start_date.replace('-', '')
        if end_date:
            end_date = end_date.replace('-', '')
        
        # 确定查询的时间范围
        if start_date and end_date:
            cache_key = f'etf_hist_{code}_{start_date}_{end_date}'
            query_start = start_date
            query_end = end_date
        else:
            cache_key = f'etf_hist_{code}_{days}'
            query_end = datetime.now().strftime('%Y%m%d')
            query_start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        # 1. 先检查内存缓存
        cached = _cache.get(cache_key, CACHE_TTL['etf_hist'])
        if cached is not None:
            return cached
        
        # 2. 从 CSV 文件加载
        csv_path = _get_csv_path('etf', code)
        csv_df = _load_from_csv(csv_path)
        
        query_start_dt = pd.to_datetime(query_start)
        query_end_dt = pd.to_datetime(query_end)
        
        # 3. 检查 CSV 数据是否满足需求
        need_download_early = False  # 需要下载更早的数据
        need_download_recent = False  # 需要下载更新的数据
        
        if csv_df.empty:
            need_download_early = True
        else:
            csv_min_date = csv_df['date'].min()
            csv_max_date = csv_df['date'].max()
            
            # 检查是否需要下载更早的数据
            if query_start_dt < csv_min_date:
                need_download_early = True
            
            # 检查是否需要下载更新的数据（允许1天的误差，因为当天可能还没收盘）
            if query_end_dt > csv_max_date + timedelta(days=1):
                need_download_recent = True
        
        # 4. 如果需要，从 API 下载数据
        if need_download_early:
            try:
                # 下载更早的数据
                download_end = (csv_df['date'].min() - timedelta(days=1)).strftime('%Y%m%d') if not csv_df.empty else query_end
                new_df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=query_start,
                    end_date=download_end,
                    adjust="qfq"
                )
                
                if not new_df.empty:
                    new_df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'change_pct', 'change_amount', 'turnover']
                    new_df['date'] = pd.to_datetime(new_df['date'])
                    numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
                    for col in numeric_cols:
                        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    csv_df = _merge_and_update_csv(csv_df, new_df, csv_path)
            except Exception as e:
                if csv_df.empty:
                    raise Exception(f"获取ETF历史数据失败: {str(e)}")
        
        if need_download_recent and not csv_df.empty:
            try:
                # 下载更新的数据
                download_start = (csv_df['date'].max() + timedelta(days=1)).strftime('%Y%m%d')
                new_df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=download_start,
                    end_date=query_end,
                    adjust="qfq"
                )
                
                if not new_df.empty:
                    new_df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'change_pct', 'change_amount', 'turnover']
                    new_df['date'] = pd.to_datetime(new_df['date'])
                    numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
                    for col in numeric_cols:
                        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    csv_df = _merge_and_update_csv(csv_df, new_df, csv_path)
            except Exception:
                pass  # 下载最新数据失败不影响使用已有数据
        
        if csv_df.empty:
            return pd.DataFrame()
        
        # 5. 按时间范围过滤数据
        result_df = csv_df[(csv_df['date'] >= query_start_dt) & (csv_df['date'] <= query_end_dt)].copy()
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        # 6. 存入内存缓存
        _cache.set(cache_key, result_df)
        
        return result_df
    except Exception as e:
        raise Exception(f"获取ETF历史数据失败: {str(e)}")


def get_index_hist_data(symbol: str, days: int = 60, start_date: str = "", end_date: str = "") -> pd.DataFrame:
    """
    获取指数历史数据（优先从CSV加载，不足则从API下载）
    
    数据加载策略：
    1. 优先从本地 CSV 文件加载数据
    2. 如果 CSV 数据不足（时间范围不够），则从网络下载
    3. 下载的数据会保存到 CSV 文件中，供下次使用
    
    Args:
        symbol: 指数代码，如"sh000001"
        days: 获取最近多少天的数据（当start_date和end_date为空时使用）
        start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD
        end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD
    
    Returns:
        包含历史数据的DataFrame
    """
    try:
        _cache = get_cache()
        
        # 标准化日期格式
        if start_date:
            start_date = start_date.replace('-', '')
        if end_date:
            end_date = end_date.replace('-', '')
        
        # 确定查询的时间范围
        if start_date and end_date:
            cache_key = f'index_hist_{symbol}_{start_date}_{end_date}'
            query_start = start_date
            query_end = end_date
        else:
            cache_key = f'index_hist_{symbol}_{days}'
            query_end = datetime.now().strftime('%Y%m%d')
            query_start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        # 1. 先检查内存缓存
        cached = _cache.get(cache_key, CACHE_TTL['index_hist'])
        if cached is not None:
            return cached
        
        # 2. 从 CSV 文件加载
        csv_path = _get_csv_path('index', symbol)
        csv_df = _load_from_csv(csv_path)
        
        query_start_dt = pd.to_datetime(query_start)
        query_end_dt = pd.to_datetime(query_end)
        
        # 3. 检查 CSV 数据是否满足需求
        need_download = False
        
        if csv_df.empty:
            need_download = True
        else:
            csv_min_date = csv_df['date'].min()
            csv_max_date = csv_df['date'].max()
            
            # 检查是否需要下载更早或更新的数据（允许1天误差）
            if query_start_dt < csv_min_date or query_end_dt > csv_max_date + timedelta(days=1):
                need_download = True
        
        # 4. 如果需要，从 API 下载数据（指数接口返回全部历史数据）
        if need_download:
            try:
                new_df = ak.stock_zh_index_daily(symbol=symbol)
                
                if not new_df.empty:
                    new_df['date'] = pd.to_datetime(new_df['date'])
                    
                    # 确保数值列为float类型
                    numeric_cols = ['open', 'close', 'high', 'low', 'volume']
                    for col in numeric_cols:
                        if col in new_df.columns:
                            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    
                    # 合并并更新 CSV
                    csv_df = _merge_and_update_csv(csv_df, new_df, csv_path)
            except Exception as e:
                # 下载失败时，如果有 CSV 数据则使用 CSV 数据
                if csv_df.empty:
                    raise Exception(f"获取指数历史数据失败: {str(e)}")
        
        if csv_df.empty:
            return pd.DataFrame()
        
        # 5. 按时间范围过滤数据
        result_df = csv_df[(csv_df['date'] >= query_start_dt) & (csv_df['date'] <= query_end_dt)].copy()
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        # 6. 存入内存缓存
        _cache.set(cache_key, result_df)
        
        return result_df
    except Exception as e:
        raise Exception(f"获取指数历史数据失败: {str(e)}")
