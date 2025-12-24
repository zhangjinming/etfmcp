"""
技术指标计算模块
提供各类技术指标的计算函数
"""

import pandas as pd
import numpy as np


def calculate_ma(data: pd.Series, period: int) -> pd.Series:
    """计算移动平均线"""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_boll(data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> dict:
    """计算布林带指标"""
    close = data['close']
    middle = calculate_ma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'bandwidth': (upper - lower) / middle * 100,  # 带宽百分比
        'percent_b': (close - lower) / (upper - lower) * 100  # %B指标
    }


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI相对强弱指标"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """计算MACD指标"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal)
    macd_hist = 2 * (dif - dea)
    
    return {
        'dif': dif,
        'dea': dea,
        'macd': macd_hist
    }


def calculate_kdj(data: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> dict:
    """计算KDJ指标"""
    low_min = data['low'].rolling(window=n).min()
    high_max = data['high'].rolling(window=n).max()
    
    rsv = (data['close'] - low_min) / (high_max - low_min) * 100
    
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return {'k': k, 'd': d, 'j': j}


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ATR平均真实波幅"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """计算OBV能量潮指标（向量化实现）"""
    close = data['close']
    volume = data['volume']
    
    # 使用向量化计算
    price_change = close.diff()
    direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    obv = (volume * direction).cumsum()
    
    return obv


def resample_to_weekly(data: pd.DataFrame) -> pd.DataFrame:
    """将日线数据转换为周线数据"""
    data = data.copy()
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    
    weekly = data.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    weekly.reset_index(inplace=True)
    return weekly


def format_indicator_summary(indicators: dict, name: str) -> str:
    """格式化技术指标摘要"""
    summary = f"=== {name} 技术指标分析 ===\n\n"
    
    if 'price_info' in indicators:
        pi = indicators['price_info']
        summary += f"【价格信息】\n"
        summary += f"  最新价: {pi.get('latest_price', 'N/A')}\n"
        summary += f"  周涨跌幅: {pi.get('weekly_change_pct', 'N/A')}%\n"
        summary += f"  月涨跌幅: {pi.get('monthly_change_pct', 'N/A')}%\n\n"
    
    if 'boll' in indicators:
        boll = indicators['boll']
        summary += f"【布林带 BOLL】\n"
        summary += f"  上轨: {boll.get('upper', 'N/A')}\n"
        summary += f"  中轨: {boll.get('middle', 'N/A')}\n"
        summary += f"  下轨: {boll.get('lower', 'N/A')}\n"
        summary += f"  带宽: {boll.get('bandwidth', 'N/A')}%\n"
        summary += f"  %B: {boll.get('percent_b', 'N/A')}%\n"
        summary += f"  信号: {boll.get('signal', 'N/A')}\n\n"
    
    if 'rsi' in indicators:
        rsi = indicators['rsi']
        summary += f"【RSI 相对强弱】\n"
        summary += f"  RSI(6): {rsi.get('rsi_6', 'N/A')}\n"
        summary += f"  RSI(12): {rsi.get('rsi_12', 'N/A')}\n"
        summary += f"  RSI(14): {rsi.get('rsi_14', 'N/A')}\n"
        summary += f"  信号: {rsi.get('signal', 'N/A')}\n\n"
    
    if 'macd' in indicators:
        macd = indicators['macd']
        summary += f"【MACD 指标】\n"
        summary += f"  DIF: {macd.get('dif', 'N/A')}\n"
        summary += f"  DEA: {macd.get('dea', 'N/A')}\n"
        summary += f"  MACD柱: {macd.get('macd', 'N/A')}\n"
        summary += f"  信号: {macd.get('signal', 'N/A')}\n\n"
    
    if 'kdj' in indicators:
        kdj = indicators['kdj']
        summary += f"【KDJ 随机指标】\n"
        summary += f"  K: {kdj.get('k', 'N/A')}\n"
        summary += f"  D: {kdj.get('d', 'N/A')}\n"
        summary += f"  J: {kdj.get('j', 'N/A')}\n"
        summary += f"  信号: {kdj.get('signal', 'N/A')}\n\n"
    
    if 'ma' in indicators:
        ma = indicators['ma']
        summary += f"【均线系统】\n"
        summary += f"  MA5: {ma.get('ma5', 'N/A')}\n"
        summary += f"  MA10: {ma.get('ma10', 'N/A')}\n"
        summary += f"  MA20: {ma.get('ma20', 'N/A')}\n"
        summary += f"  MA60: {ma.get('ma60', 'N/A')}\n"
        summary += f"  趋势: {ma.get('trend', 'N/A')}\n\n"
    
    if 'volume' in indicators:
        vol = indicators['volume']
        summary += f"【成交量分析】\n"
        summary += f"  当前成交量: {vol.get('current', 'N/A')}\n"
        summary += f"  5日均量: {vol.get('ma5', 'N/A')}\n"
        summary += f"  量比: {vol.get('volume_ratio', 'N/A')}\n\n"
    
    return summary


def get_indicator_signals(indicators: dict) -> dict:
    """生成技术指标信号汇总"""
    signals = {
        'bullish': [],  # 看涨信号
        'bearish': [],  # 看跌信号
        'neutral': [],  # 中性信号
        'overall': ''   # 综合判断
    }
    
    # BOLL信号
    if 'boll' in indicators:
        boll = indicators['boll']
        pb = boll.get('percent_b', 50)
        if pb is not None:
            if pb < 20:
                signals['bullish'].append('BOLL: 价格接近下轨，可能超卖')
            elif pb > 80:
                signals['bearish'].append('BOLL: 价格接近上轨，可能超买')
            else:
                signals['neutral'].append('BOLL: 价格在布林带中间区域')
    
    # RSI信号
    if 'rsi' in indicators:
        rsi = indicators['rsi']
        rsi_14 = rsi.get('rsi_14', 50)
        if rsi_14 is not None:
            if rsi_14 < 30:
                signals['bullish'].append(f'RSI({rsi_14:.1f}): 超卖区域，可能反弹')
            elif rsi_14 > 70:
                signals['bearish'].append(f'RSI({rsi_14:.1f}): 超买区域，可能回调')
            else:
                signals['neutral'].append(f'RSI({rsi_14:.1f}): 中性区域')
    
    # MACD信号
    if 'macd' in indicators:
        macd = indicators['macd']
        dif = macd.get('dif', 0)
        dea = macd.get('dea', 0)
        if dif is not None and dea is not None:
            if dif > dea and dif > 0:
                signals['bullish'].append('MACD: DIF在DEA上方且为正，多头强势')
            elif dif < dea and dif < 0:
                signals['bearish'].append('MACD: DIF在DEA下方且为负，空头强势')
            elif dif > dea:
                signals['bullish'].append('MACD: 金叉形成，看涨信号')
            else:
                signals['bearish'].append('MACD: 死叉形成，看跌信号')
    
    # KDJ信号
    if 'kdj' in indicators:
        kdj = indicators['kdj']
        k = kdj.get('k', 50)
        d = kdj.get('d', 50)
        if k is not None and d is not None:
            if k < 20 and d < 20:
                signals['bullish'].append('KDJ: 超卖区域，可能反弹')
            elif k > 80 and d > 80:
                signals['bearish'].append('KDJ: 超买区域，可能回调')
            elif k > d:
                signals['bullish'].append('KDJ: K线在D线上方，短期看涨')
            else:
                signals['bearish'].append('KDJ: K线在D线下方，短期看跌')
    
    # 综合判断
    bull_count = len(signals['bullish'])
    bear_count = len(signals['bearish'])
    
    if bull_count > bear_count + 1:
        signals['overall'] = '综合看涨'
    elif bear_count > bull_count + 1:
        signals['overall'] = '综合看跌'
    else:
        signals['overall'] = '综合中性/震荡'
    
    return signals


def calculate_period_score(df: pd.DataFrame) -> dict:
    """计算单个周期的技术指标评分"""
    if len(df) < 20:
        return None
    
    latest_price = df['close'].iloc[-1]
    
    # 均线
    ma5 = calculate_ma(df['close'], 5).iloc[-1]
    ma10 = calculate_ma(df['close'], 10).iloc[-1]
    ma20 = calculate_ma(df['close'], 20).iloc[-1]
    ma60 = calculate_ma(df['close'], min(60, len(df)-1)).iloc[-1] if len(df) > 60 else ma20
    
    # MACD
    macd = calculate_macd(df['close'])
    dif = macd['dif'].iloc[-1]
    dea = macd['dea'].iloc[-1]
    
    # RSI
    rsi_14 = calculate_rsi(df['close'], 14).iloc[-1]
    
    # BOLL
    boll = calculate_boll(df)
    percent_b = boll['percent_b'].iloc[-1]
    
    # 成交量
    vol_ma5 = calculate_ma(df['volume'], 5).iloc[-1]
    vol_ma20 = calculate_ma(df['volume'], min(20, len(df)-1)).iloc[-1] if len(df) > 20 else vol_ma5
    current_vol = df['volume'].iloc[-1]
    volume_ratio = current_vol / vol_ma5 if vol_ma5 > 0 else 1
    vol_trend = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1
    
    # 评分
    score = 0
    details = []
    
    # BOLL (35分)
    if percent_b < 10:
        score += 35
        details.append(f"BOLL严重超卖({percent_b:.1f}%)")
    elif percent_b < 20:
        score += 25
        details.append(f"BOLL接近下轨({percent_b:.1f}%)")
    elif percent_b < 35:
        score += 15
        details.append(f"BOLL偏下轨({percent_b:.1f}%)")
    elif percent_b > 90:
        score -= 35
        details.append(f"BOLL严重超买({percent_b:.1f}%)")
    elif percent_b > 80:
        score -= 25
        details.append(f"BOLL接近上轨({percent_b:.1f}%)")
    elif percent_b > 65:
        score -= 15
        details.append(f"BOLL偏上轨({percent_b:.1f}%)")
    
    # 成交量 (20分)
    if volume_ratio > 2.0 and latest_price > ma5:
        score += 20
    elif volume_ratio > 1.5 and latest_price > ma5:
        score += 15
    elif volume_ratio > 2.0 and latest_price < ma5:
        score -= 20
    elif volume_ratio > 1.5 and latest_price < ma5:
        score -= 15
    elif vol_trend > 1.2:
        score += 5
    elif vol_trend < 0.8:
        score -= 5
    
    # RSI (15分)
    if rsi_14 < 20:
        score += 15
        details.append(f"RSI严重超卖({rsi_14:.1f})")
    elif rsi_14 < 30:
        score += 10
        details.append(f"RSI超卖({rsi_14:.1f})")
    elif rsi_14 > 80:
        score -= 15
        details.append(f"RSI严重超买({rsi_14:.1f})")
    elif rsi_14 > 70:
        score -= 10
        details.append(f"RSI超买({rsi_14:.1f})")
    elif rsi_14 > 50:
        score += 5
    else:
        score -= 5
    
    # MACD (15分)
    if dif > dea and dif > 0:
        score += 15
        details.append("MACD零轴上金叉")
    elif dif > dea and dif < 0:
        score += 8
        details.append("MACD零轴下金叉")
    elif dif < dea and dif < 0:
        score -= 15
        details.append("MACD零轴下死叉")
    elif dif < dea and dif > 0:
        score -= 8
        details.append("MACD零轴上死叉")
    
    # 均线 (15分)
    if latest_price > ma5 > ma10 > ma20 > ma60:
        score += 15
        details.append("均线多头排列")
    elif latest_price > ma5 > ma10 > ma20:
        score += 10
        details.append("短期多头")
    elif latest_price < ma5 < ma10 < ma20 < ma60:
        score -= 15
        details.append("均线空头排列")
    elif latest_price < ma5 < ma10 < ma20:
        score -= 10
        details.append("短期空头")
    else:
        details.append("均线交织")
    
    return {
        'score': score,
        'details': details,
        'rsi': rsi_14,
        'percent_b': percent_b,
        'dif': dif,
        'dea': dea,
        'ma5': ma5,
        'ma10': ma10,
        'ma20': ma20,
        'volume_ratio': volume_ratio
    }


def analyze_historical_indicators(df: pd.DataFrame, weeks: int) -> dict:
    """分析历史周期内的技术指标统计"""
    if len(df) < weeks:
        weeks = len(df)
    
    # 取指定周数的数据
    period_df = df.tail(weeks).copy()
    
    # 计算整个周期的指标序列
    rsi_series = calculate_rsi(period_df['close'], 14)
    boll = calculate_boll(period_df)
    macd = calculate_macd(period_df['close'])
    
    # 统计金叉死叉次数
    dif_series = macd['dif']
    dea_series = macd['dea']
    cross_up = 0  # 金叉次数
    cross_down = 0  # 死叉次数
    for i in range(1, len(dif_series)):
        if dif_series.iloc[i] > dea_series.iloc[i] and dif_series.iloc[i-1] <= dea_series.iloc[i-1]:
            cross_up += 1
        elif dif_series.iloc[i] < dea_series.iloc[i] and dif_series.iloc[i-1] >= dea_series.iloc[i-1]:
            cross_down += 1
    
    # RSI统计
    rsi_valid = rsi_series.dropna()
    rsi_oversold_count = (rsi_valid < 30).sum()  # 超卖次数
    rsi_overbought_count = (rsi_valid > 70).sum()  # 超买次数
    rsi_avg = rsi_valid.mean()
    rsi_min = rsi_valid.min()
    rsi_max = rsi_valid.max()
    
    # BOLL %B统计
    pb_series = boll['percent_b'].dropna()
    pb_near_lower = (pb_series < 20).sum()  # 接近下轨次数
    pb_near_upper = (pb_series > 80).sum()  # 接近上轨次数
    pb_avg = pb_series.mean()
    pb_min = pb_series.min()
    pb_max = pb_series.max()
    
    # 价格涨跌幅
    start_price = period_df['close'].iloc[0]
    end_price = period_df['close'].iloc[-1]
    total_change = (end_price - start_price) / start_price * 100
    
    # 最大回撤
    cummax = period_df['close'].cummax()
    drawdown = (period_df['close'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # 最大涨幅（从最低点算起）
    cummin = period_df['close'].cummin()
    rally = (period_df['close'] - cummin) / cummin * 100
    max_rally = rally.max()
    
    # 周度涨跌统计
    weekly_changes = period_df['close'].pct_change() * 100
    up_weeks = (weekly_changes > 0).sum()
    down_weeks = (weekly_changes < 0).sum()
    
    return {
        'weeks': weeks,
        'total_change': round(total_change, 2),
        'max_drawdown': round(max_drawdown, 2),
        'max_rally': round(max_rally, 2),
        'up_weeks': int(up_weeks),
        'down_weeks': int(down_weeks),
        'rsi_avg': round(rsi_avg, 1),
        'rsi_min': round(rsi_min, 1),
        'rsi_max': round(rsi_max, 1),
        'rsi_oversold_count': int(rsi_oversold_count),
        'rsi_overbought_count': int(rsi_overbought_count),
        'pb_avg': round(pb_avg, 1),
        'pb_min': round(pb_min, 1),
        'pb_max': round(pb_max, 1),
        'pb_near_lower': int(pb_near_lower),
        'pb_near_upper': int(pb_near_upper),
        'macd_cross_up': cross_up,
        'macd_cross_down': cross_down
    }


def get_period_trend_judgment(stats: dict) -> tuple:
    """根据历史统计给出周期趋势判断和评分"""
    score = 0
    judgments = []
    
    # 涨跌幅评分 (30分)
    change = stats['total_change']
    if change > 30:
        score += 30
        judgments.append(f"大幅上涨{change}%")
    elif change > 15:
        score += 20
        judgments.append(f"明显上涨{change}%")
    elif change > 5:
        score += 10
        judgments.append(f"小幅上涨{change}%")
    elif change > -5:
        judgments.append(f"横盘整理{change}%")
    elif change > -15:
        score -= 10
        judgments.append(f"小幅下跌{change}%")
    elif change > -30:
        score -= 20
        judgments.append(f"明显下跌{change}%")
    else:
        score -= 30
        judgments.append(f"大幅下跌{change}%")
    
    # 回撤风险评分 (20分)
    drawdown = abs(stats['max_drawdown'])
    if drawdown < 5:
        score += 20
        judgments.append(f"回撤极小({drawdown}%)")
    elif drawdown < 10:
        score += 10
        judgments.append(f"回撤可控({drawdown}%)")
    elif drawdown < 20:
        judgments.append(f"回撤适中({drawdown}%)")
    elif drawdown < 30:
        score -= 10
        judgments.append(f"回撤较大({drawdown}%)")
    else:
        score -= 20
        judgments.append(f"回撤严重({drawdown}%)")
    
    # 胜率评分 (15分)
    total_weeks = stats['up_weeks'] + stats['down_weeks']
    if total_weeks > 0:
        win_rate = stats['up_weeks'] / total_weeks * 100
        if win_rate > 65:
            score += 15
            judgments.append(f"周胜率高({win_rate:.0f}%)")
        elif win_rate > 55:
            score += 8
            judgments.append(f"周胜率偏高({win_rate:.0f}%)")
        elif win_rate > 45:
            judgments.append(f"周胜率平衡({win_rate:.0f}%)")
        elif win_rate > 35:
            score -= 8
            judgments.append(f"周胜率偏低({win_rate:.0f}%)")
        else:
            score -= 15
            judgments.append(f"周胜率低({win_rate:.0f}%)")
    
    # RSI历史表现 (15分)
    if stats['rsi_oversold_count'] > stats['rsi_overbought_count'] + 2:
        score += 15
        judgments.append(f"RSI多次超卖({stats['rsi_oversold_count']}次)")
    elif stats['rsi_overbought_count'] > stats['rsi_oversold_count'] + 2:
        score -= 15
        judgments.append(f"RSI多次超买({stats['rsi_overbought_count']}次)")
    elif stats['rsi_avg'] > 55:
        score += 8
        judgments.append(f"RSI均值偏强({stats['rsi_avg']})")
    elif stats['rsi_avg'] < 45:
        score -= 8
        judgments.append(f"RSI均值偏弱({stats['rsi_avg']})")
    
    # MACD金叉死叉 (10分)
    if stats['macd_cross_up'] > stats['macd_cross_down'] + 1:
        score += 10
        judgments.append(f"MACD金叉多({stats['macd_cross_up']}次金叉)")
    elif stats['macd_cross_down'] > stats['macd_cross_up'] + 1:
        score -= 10
        judgments.append(f"MACD死叉多({stats['macd_cross_down']}次死叉)")
    
    # BOLL触轨情况 (10分)
    if stats['pb_near_lower'] > stats['pb_near_upper'] + 2:
        score += 10
        judgments.append(f"多次触及下轨({stats['pb_near_lower']}次)")
    elif stats['pb_near_upper'] > stats['pb_near_lower'] + 2:
        score -= 10
        judgments.append(f"多次触及上轨({stats['pb_near_upper']}次)")
    
    return score, judgments
