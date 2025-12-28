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
    """
    计算RSI相对强弱指标（使用Wilder平滑方法，与主流交易软件一致）
    
    Wilder平滑法：使用 EMA(alpha=1/period) 而非 SMA
    这与同花顺、通达信等软件的计算方式一致
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 使用 Wilder 平滑（等价于 EMA，alpha = 1/period）
    # ewm(alpha=1/period) 等价于 ewm(span=2*period-1, adjust=False) 的近似
    # 但更准确的 Wilder 方法是 ewm(com=period-1, adjust=False)
    avg_gain = gain.ewm(com=period-1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


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
    
    # 均线趋势信号
    if 'ma' in indicators:
        ma = indicators['ma']
        trend = ma.get('trend', '')
        if '多头' in trend or '上升' in trend:
            signals['bullish'].append(f'均线: {trend}')
        elif '空头' in trend or '下降' in trend:
            signals['bearish'].append(f'均线: {trend}')
        else:
            signals['neutral'].append(f'均线: {trend}')
    
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


# ========== 波段交易策略相关指标 ==========

def analyze_weekly_trend(df: pd.DataFrame) -> dict:
    """
    分析周线趋势（波段策略核心）
    使用 MA20（生命线）和 MA60（牛熊分界线）判断中期趋势
    
    Args:
        df: 周线数据DataFrame
    
    Returns:
        趋势分析结果字典
    """
    if len(df) < 60:
        return {'error': '数据不足60周，无法进行完整趋势分析'}
    
    close = df['close']
    latest_price = close.iloc[-1]
    
    # 计算关键均线
    ma20 = calculate_ma(close, 20)
    ma60 = calculate_ma(close, 60)
    ma5 = calculate_ma(close, 5)
    
    ma20_current = ma20.iloc[-1]
    ma60_current = ma60.iloc[-1]
    ma5_current = ma5.iloc[-1]
    
    # 均线斜率（判断方向）
    ma20_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100 if len(ma20) >= 5 else 0
    ma60_slope = (ma60.iloc[-1] - ma60.iloc[-5]) / ma60.iloc[-5] * 100 if len(ma60) >= 5 else 0
    
    # MACD（周线级别）
    macd = calculate_macd(close)
    dif = macd['dif'].iloc[-1]
    dea = macd['dea'].iloc[-1]
    macd_hist = macd['macd'].iloc[-1]
    
    # 判断趋势类型
    trend_type = "震荡"
    trend_strength = 0  # -100 到 100
    trend_signals = []
    
    # 1. 价格与均线位置关系
    price_above_ma20 = latest_price > ma20_current
    price_above_ma60 = latest_price > ma60_current
    ma20_above_ma60 = ma20_current > ma60_current
    
    # 2. 多头趋势判断
    if price_above_ma20 and price_above_ma60 and ma20_above_ma60:
        if ma20_slope > 0.5 and ma60_slope > 0:
            trend_type = "强势多头"
            trend_strength = 80
            trend_signals.append("价格在MA20和MA60上方")
            trend_signals.append("MA20在MA60上方，均线多头排列")
            trend_signals.append(f"MA20向上发散(斜率{ma20_slope:.2f}%)")
        else:
            trend_type = "多头趋势"
            trend_strength = 60
            trend_signals.append("均线多头排列")
    
    # 3. 空头趋势判断
    elif not price_above_ma20 and not price_above_ma60 and not ma20_above_ma60:
        if ma20_slope < -0.5 and ma60_slope < 0:
            trend_type = "强势空头"
            trend_strength = -80
            trend_signals.append("价格在MA20和MA60下方")
            trend_signals.append("MA20在MA60下方，均线空头排列")
            trend_signals.append(f"MA20向下发散(斜率{ma20_slope:.2f}%)")
        else:
            trend_type = "空头趋势"
            trend_strength = -60
            trend_signals.append("均线空头排列")
    
    # 4. 震荡/转势判断
    elif price_above_ma20 and not ma20_above_ma60:
        trend_type = "底部反转中"
        trend_strength = 30
        trend_signals.append("价格站上MA20但MA20仍在MA60下方")
        trend_signals.append("可能处于底部反转初期")
    elif not price_above_ma20 and ma20_above_ma60:
        trend_type = "顶部回调中"
        trend_strength = -30
        trend_signals.append("价格跌破MA20但MA20仍在MA60上方")
        trend_signals.append("可能处于顶部回调期")
    else:
        trend_type = "震荡整理"
        trend_strength = 0
        trend_signals.append("均线交织，方向不明")
    
    # 5. MACD趋势确认
    macd_position = "零轴上方" if dif > 0 else "零轴下方"
    macd_cross = "金叉" if dif > dea else "死叉"
    
    if dif > 0 and dif > dea:
        trend_strength = min(100, trend_strength + 15)
        trend_signals.append(f"MACD{macd_position}{macd_cross}，动能向上")
    elif dif < 0 and dif < dea:
        trend_strength = max(-100, trend_strength - 15)
        trend_signals.append(f"MACD{macd_position}{macd_cross}，动能向下")
    else:
        trend_signals.append(f"MACD{macd_position}{macd_cross}")
    
    # 6. 检测背离
    divergence = detect_macd_divergence(df, 'weekly')
    if divergence['type'] != 'none':
        trend_signals.append(f"周线MACD{divergence['type']}，{divergence['description']}")
    
    # 操作建议
    if trend_strength >= 60:
        operation = "只做多，等待日线回调买入机会"
    elif trend_strength >= 30:
        operation = "偏多操作，轻仓试探性买入"
    elif trend_strength <= -60:
        operation = "空仓观望，等待止跌信号"
    elif trend_strength <= -30:
        operation = "减仓或观望，谨慎操作"
    else:
        operation = "观望等待，方向不明确"
    
    return {
        'trend_type': trend_type,
        'trend_strength': trend_strength,
        'signals': trend_signals,
        'operation': operation,
        'ma5': round(ma5_current, 4),
        'ma20': round(ma20_current, 4),
        'ma60': round(ma60_current, 4),
        'ma20_slope': round(ma20_slope, 2),
        'ma60_slope': round(ma60_slope, 2),
        'price_vs_ma20': round((latest_price - ma20_current) / ma20_current * 100, 2),
        'price_vs_ma60': round((latest_price - ma60_current) / ma60_current * 100, 2),
        'macd_dif': round(dif, 4),
        'macd_dea': round(dea, 4),
        'macd_hist': round(macd_hist, 4),
        'macd_position': macd_position,
        'macd_cross': macd_cross,
        'divergence': divergence
    }


def detect_macd_divergence(df: pd.DataFrame, period_type: str = 'daily') -> dict:
    """
    检测MACD背离
    
    Args:
        df: 价格数据DataFrame
        period_type: 'daily' 或 'weekly'
    
    Returns:
        背离信息字典
    """
    if len(df) < 30:
        return {'type': 'none', 'description': '数据不足'}
    
    close = df['close']
    macd = calculate_macd(close)
    dif = macd['dif']
    
    # 寻找近期的价格高点/低点和对应的DIF值
    lookback = 20 if period_type == 'daily' else 12
    
    recent_close = close.tail(lookback)
    recent_dif = dif.tail(lookback)
    
    # 找局部极值点
    price_highs = []
    price_lows = []
    dif_at_highs = []
    dif_at_lows = []
    
    for i in range(2, len(recent_close) - 2):
        # 局部高点
        if (recent_close.iloc[i] > recent_close.iloc[i-1] and 
            recent_close.iloc[i] > recent_close.iloc[i-2] and
            recent_close.iloc[i] > recent_close.iloc[i+1] and 
            recent_close.iloc[i] > recent_close.iloc[i+2]):
            price_highs.append(recent_close.iloc[i])
            dif_at_highs.append(recent_dif.iloc[i])
        
        # 局部低点
        if (recent_close.iloc[i] < recent_close.iloc[i-1] and 
            recent_close.iloc[i] < recent_close.iloc[i-2] and
            recent_close.iloc[i] < recent_close.iloc[i+1] and 
            recent_close.iloc[i] < recent_close.iloc[i+2]):
            price_lows.append(recent_close.iloc[i])
            dif_at_lows.append(recent_dif.iloc[i])
    
    # 检测顶背离：价格新高，DIF未新高
    if len(price_highs) >= 2 and len(dif_at_highs) >= 2:
        if price_highs[-1] > price_highs[-2] and dif_at_highs[-1] < dif_at_highs[-2]:
            return {
                'type': '顶背离',
                'description': '价格创新高但MACD未创新高，可能见顶回调',
                'strength': 'strong' if dif_at_highs[-1] < dif_at_highs[-2] * 0.8 else 'weak'
            }
    
    # 检测底背离：价格新低，DIF未新低
    if len(price_lows) >= 2 and len(dif_at_lows) >= 2:
        if price_lows[-1] < price_lows[-2] and dif_at_lows[-1] > dif_at_lows[-2]:
            return {
                'type': '底背离',
                'description': '价格创新低但MACD未创新低，可能见底反弹',
                'strength': 'strong' if dif_at_lows[-1] > dif_at_lows[-2] * 1.2 else 'weak'
            }
    
    return {'type': 'none', 'description': '未检测到明显背离'}


def analyze_daily_buy_signals(df: pd.DataFrame, weekly_support: float = None, weekly_ma20: float = None) -> dict:
    """
    分析日线买入信号（在周线多头趋势下使用）
    
    Args:
        df: 日线数据DataFrame
        weekly_support: 周线关键支撑位
        weekly_ma20: 周线MA20值
    
    Returns:
        买入信号分析结果
    """
    if len(df) < 60:
        return {'error': '数据不足'}
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    latest_price = close.iloc[-1]
    
    signals = []
    signal_strength = 0  # 0-100分
    
    # 1. 计算日线指标
    ma5 = calculate_ma(close, 5).iloc[-1]
    ma10 = calculate_ma(close, 10).iloc[-1]
    ma20 = calculate_ma(close, 20).iloc[-1]
    ma50 = calculate_ma(close, 50).iloc[-1] if len(df) >= 50 else ma20
    ma60 = calculate_ma(close, 60).iloc[-1] if len(df) >= 60 else ma50
    ema50 = calculate_ema(close, 50).iloc[-1]
    
    # 2. MACD
    macd = calculate_macd(close)
    dif = macd['dif'].iloc[-1]
    dea = macd['dea'].iloc[-1]
    prev_dif = macd['dif'].iloc[-2]
    prev_dea = macd['dea'].iloc[-2]
    
    # 3. KDJ
    kdj = calculate_kdj(df, 9, 3, 3)
    k = kdj['k'].iloc[-1]
    d = kdj['d'].iloc[-1]
    j = kdj['j'].iloc[-1]
    prev_k = kdj['k'].iloc[-2]
    prev_d = kdj['d'].iloc[-2]
    
    # 4. RSI
    rsi_14 = calculate_rsi(close, 14).iloc[-1]
    
    # 5. BOLL
    boll = calculate_boll(df)
    boll_lower = boll['lower'].iloc[-1]
    boll_middle = boll['middle'].iloc[-1]
    percent_b = boll['percent_b'].iloc[-1]
    
    # 6. 成交量
    vol_ma5 = calculate_ma(volume, 5).iloc[-1]
    vol_ma20 = calculate_ma(volume, 20).iloc[-1]
    current_vol = volume.iloc[-1]
    volume_ratio = current_vol / vol_ma5 if vol_ma5 > 0 else 1
    
    # ========== 买入信号检测 ==========
    
    # 信号1: 位置 - 回调至关键支撑位
    position_signal = False
    if weekly_support and abs(latest_price - weekly_support) / weekly_support < 0.02:
        signals.append(f"价格接近周线关键支撑位{weekly_support:.4f}")
        signal_strength += 20
        position_signal = True
    
    if weekly_ma20 and abs(latest_price - weekly_ma20) / weekly_ma20 < 0.02:
        signals.append(f"价格回踩周线MA20({weekly_ma20:.4f})")
        signal_strength += 15
        position_signal = True
    
    # 回踩日线EMA50/60
    if abs(latest_price - ema50) / ema50 < 0.015:
        signals.append(f"价格回踩日线EMA50({ema50:.4f})")
        signal_strength += 10
        position_signal = True
    
    # 信号2: MACD底背离
    divergence = detect_macd_divergence(df, 'daily')
    if divergence['type'] == '底背离':
        signals.append(f"日线MACD底背离: {divergence['description']}")
        signal_strength += 25 if divergence['strength'] == 'strong' else 15
    
    # MACD金叉（DIF拐头向上）
    if dif > dea and prev_dif <= prev_dea:
        signals.append("日线MACD刚形成金叉")
        signal_strength += 15
    elif dif > prev_dif and dif < dea:
        signals.append("日线MACD DIF拐头向上，金叉在即")
        signal_strength += 10
    
    # 信号3: KDJ超卖金叉
    kdj_oversold = j < 20
    kdj_golden_cross = k > d and prev_k <= prev_d
    
    if kdj_oversold:
        signals.append(f"KDJ进入超卖区(J={j:.1f})")
        signal_strength += 10
    
    if kdj_golden_cross and j < 30:
        signals.append("KDJ超卖区金叉，买入信号")
        signal_strength += 20
    elif kdj_golden_cross:
        signals.append("KDJ形成金叉")
        signal_strength += 10
    
    # 信号4: RSI超卖
    if rsi_14 < 30:
        signals.append(f"RSI超卖({rsi_14:.1f})")
        signal_strength += 15
    elif rsi_14 < 40:
        signals.append(f"RSI偏弱({rsi_14:.1f})")
        signal_strength += 5
    
    # 信号5: BOLL下轨
    if percent_b < 10:
        signals.append(f"价格触及BOLL下轨(%%B={percent_b:.1f}%)")
        signal_strength += 15
    elif percent_b < 20:
        signals.append(f"价格接近BOLL下轨(%%B={percent_b:.1f}%)")
        signal_strength += 10
    
    # 信号6: 成交量萎缩（卖盘枯竭）
    if volume_ratio < 0.6:
        signals.append(f"成交量显著萎缩(量比{volume_ratio:.2f})，卖盘枯竭")
        signal_strength += 10
    elif volume_ratio < 0.8:
        signals.append(f"成交量萎缩(量比{volume_ratio:.2f})")
        signal_strength += 5
    
    # 信号7: K线形态（简单检测）
    candle_signal = detect_bullish_candle_pattern(df)
    if candle_signal:
        signals.append(candle_signal)
        signal_strength += 10
    
    # 综合判断
    signal_strength = min(100, signal_strength)
    
    if signal_strength >= 70:
        recommendation = "强烈买入信号，可考虑建仓40%"
    elif signal_strength >= 50:
        recommendation = "较强买入信号，可考虑轻仓试探"
    elif signal_strength >= 30:
        recommendation = "信号一般，建议继续观察"
    else:
        recommendation = "买入信号不足，继续等待"
    
    return {
        'signal_strength': signal_strength,
        'signals': signals,
        'recommendation': recommendation,
        'indicators': {
            'ma5': round(ma5, 4),
            'ma10': round(ma10, 4),
            'ma20': round(ma20, 4),
            'ma60': round(ma60, 4),
            'ema50': round(ema50, 4),
            'macd_dif': round(dif, 4),
            'macd_dea': round(dea, 4),
            'kdj_k': round(k, 1),
            'kdj_d': round(d, 1),
            'kdj_j': round(j, 1),
            'rsi_14': round(rsi_14, 1),
            'boll_lower': round(boll_lower, 4),
            'percent_b': round(percent_b, 1),
            'volume_ratio': round(volume_ratio, 2)
        },
        'divergence': divergence
    }


def analyze_daily_sell_signals(df: pd.DataFrame, entry_price: float = None, weekly_resistance: float = None) -> dict:
    """
    分析日线卖出信号
    
    Args:
        df: 日线数据DataFrame
        entry_price: 买入价格（用于计算止盈止损）
        weekly_resistance: 周线阻力位
    
    Returns:
        卖出信号分析结果
    """
    if len(df) < 30:
        return {'error': '数据不足'}
    
    close = df['close']
    latest_price = close.iloc[-1]
    
    signals = []
    signal_strength = 0  # 0-100分
    
    # 计算指标
    ma5 = calculate_ma(close, 5).iloc[-1]
    ma10 = calculate_ma(close, 10).iloc[-1]
    
    macd = calculate_macd(close)
    dif = macd['dif'].iloc[-1]
    dea = macd['dea'].iloc[-1]
    prev_dif = macd['dif'].iloc[-2]
    prev_dea = macd['dea'].iloc[-2]
    
    kdj = calculate_kdj(df, 9, 3, 3)
    k = kdj['k'].iloc[-1]
    d = kdj['d'].iloc[-1]
    j = kdj['j'].iloc[-1]
    prev_k = kdj['k'].iloc[-2]
    prev_d = kdj['d'].iloc[-2]
    
    rsi_14 = calculate_rsi(close, 14).iloc[-1]
    
    boll = calculate_boll(df)
    percent_b = boll['percent_b'].iloc[-1]
    
    # ========== 卖出信号检测 ==========
    
    # 信号1: 到达阻力位
    if weekly_resistance and latest_price >= weekly_resistance * 0.98:
        signals.append(f"价格接近周线阻力位{weekly_resistance:.4f}")
        signal_strength += 20
    
    # 信号2: MACD顶背离
    divergence = detect_macd_divergence(df, 'daily')
    if divergence['type'] == '顶背离':
        signals.append(f"日线MACD顶背离: {divergence['description']}")
        signal_strength += 25 if divergence['strength'] == 'strong' else 15
    
    # MACD死叉
    if dif < dea and prev_dif >= prev_dea:
        signals.append("日线MACD形成死叉")
        signal_strength += 15
    elif dif < prev_dif and dif > dea:
        signals.append("日线MACD DIF拐头向下")
        signal_strength += 10
    
    # 信号3: KDJ超买死叉
    if j > 80:
        signals.append(f"KDJ进入超买区(J={j:.1f})")
        signal_strength += 10
    
    if k < d and prev_k >= prev_d and j > 70:
        signals.append("KDJ超买区死叉，卖出信号")
        signal_strength += 20
    elif k < d and prev_k >= prev_d:
        signals.append("KDJ形成死叉")
        signal_strength += 10
    
    # 信号4: RSI超买
    if rsi_14 > 80:
        signals.append(f"RSI严重超买({rsi_14:.1f})")
        signal_strength += 15
    elif rsi_14 > 70:
        signals.append(f"RSI超买({rsi_14:.1f})")
        signal_strength += 10
    
    # 信号5: BOLL上轨
    if percent_b > 95:
        signals.append(f"价格触及BOLL上轨(%%B={percent_b:.1f}%)")
        signal_strength += 15
    elif percent_b > 80:
        signals.append(f"价格接近BOLL上轨(%%B={percent_b:.1f}%)")
        signal_strength += 10
    
    # 信号6: 跌破短期均线
    if latest_price < ma5:
        signals.append(f"价格跌破MA5({ma5:.4f})")
        signal_strength += 10
    if latest_price < ma10:
        signals.append(f"价格跌破MA10({ma10:.4f})")
        signal_strength += 10
    
    # 止盈止损建议
    stop_loss = None
    take_profit = None
    
    if entry_price:
        profit_pct = (latest_price - entry_price) / entry_price * 100
        
        # 硬性止损: -4%
        stop_loss = round(entry_price * 0.96, 4)
        
        # 动态止盈
        if profit_pct >= 15:
            # 盈利15%以上，止盈位上移到成本价+10%
            take_profit = round(entry_price * 1.10, 4)
            signals.append(f"盈利{profit_pct:.1f}%，建议将止盈位上移至{take_profit}")
        elif profit_pct >= 10:
            # 盈利10%以上，止盈位上移到成本价
            take_profit = entry_price
            signals.append(f"盈利{profit_pct:.1f}%，建议将止损位上移至成本价")
    
    signal_strength = min(100, signal_strength)
    
    if signal_strength >= 70:
        recommendation = "强烈卖出信号，建议分批止盈"
    elif signal_strength >= 50:
        recommendation = "较强卖出信号，可考虑减仓"
    elif signal_strength >= 30:
        recommendation = "信号一般，可设置移动止盈"
    else:
        recommendation = "卖出信号不足，可继续持有"
    
    return {
        'signal_strength': signal_strength,
        'signals': signals,
        'recommendation': recommendation,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'divergence': divergence,
        'indicators': {
            'ma5': round(ma5, 4),
            'ma10': round(ma10, 4),
            'macd_dif': round(dif, 4),
            'macd_dea': round(dea, 4),
            'kdj_k': round(k, 1),
            'kdj_d': round(d, 1),
            'kdj_j': round(j, 1),
            'rsi_14': round(rsi_14, 1),
            'percent_b': round(percent_b, 1)
        }
    }


def detect_bullish_candle_pattern(df: pd.DataFrame) -> str:
    """
    检测看涨K线形态
    
    Args:
        df: 日线数据DataFrame
    
    Returns:
        形态描述，无形态返回None
    """
    if len(df) < 3:
        return None
    
    # 最近3根K线
    c1 = df.iloc[-3]  # 前2根
    c2 = df.iloc[-2]  # 前1根
    c3 = df.iloc[-1]  # 最新
    
    # 锤头线（下影线长，实体小，在下跌趋势末端）
    body = abs(c3['close'] - c3['open'])
    lower_shadow = min(c3['open'], c3['close']) - c3['low']
    upper_shadow = c3['high'] - max(c3['open'], c3['close'])
    
    if lower_shadow > body * 2 and upper_shadow < body * 0.5:
        return "出现锤头线，可能见底反转"
    
    # 看涨吞没
    if (c2['close'] < c2['open'] and  # 前一根阴线
        c3['close'] > c3['open'] and  # 当前阳线
        c3['open'] < c2['close'] and  # 开盘低于前收
        c3['close'] > c2['open']):    # 收盘高于前开
        return "出现看涨吞没形态"
    
    # 早晨之星（简化版）
    if (c1['close'] < c1['open'] and  # 第一根阴线
        abs(c2['close'] - c2['open']) < (c1['high'] - c1['low']) * 0.3 and  # 第二根小实体
        c3['close'] > c3['open'] and  # 第三根阳线
        c3['close'] > (c1['open'] + c1['close']) / 2):  # 收盘超过第一根中点
        return "出现早晨之星形态，反转信号"
    
    return None


def find_support_resistance(df: pd.DataFrame, lookback: int = 52) -> dict:
    """
    寻找关键支撑阻力位
    
    Args:
        df: 价格数据DataFrame
        lookback: 回看周期数
    
    Returns:
        支撑阻力位字典
    """
    if len(df) < lookback:
        lookback = len(df)
    
    recent_df = df.tail(lookback)
    close = recent_df['close']
    high = recent_df['high']
    low = recent_df['low']
    latest_price = close.iloc[-1]
    
    # 找局部高点和低点
    highs = []
    lows = []
    
    for i in range(2, len(recent_df) - 2):
        # 局部高点
        if (high.iloc[i] > high.iloc[i-1] and 
            high.iloc[i] > high.iloc[i-2] and
            high.iloc[i] > high.iloc[i+1] and 
            high.iloc[i] > high.iloc[i+2]):
            highs.append(high.iloc[i])
        
        # 局部低点
        if (low.iloc[i] < low.iloc[i-1] and 
            low.iloc[i] < low.iloc[i-2] and
            low.iloc[i] < low.iloc[i+1] and 
            low.iloc[i] < low.iloc[i+2]):
            lows.append(low.iloc[i])
    
    # 找最近的支撑位（低于当前价格的高点）
    supports = [l for l in lows if l < latest_price]
    supports.sort(reverse=True)
    
    # 找最近的阻力位（高于当前价格的高点）
    resistances = [h for h in highs if h > latest_price]
    resistances.sort()
    
    # 添加均线作为动态支撑阻力
    ma20 = calculate_ma(close, 20).iloc[-1]
    ma60 = calculate_ma(close, min(60, len(close)-1)).iloc[-1] if len(close) > 60 else ma20
    
    return {
        'supports': supports[:3] if supports else [low.min()],
        'resistances': resistances[:3] if resistances else [high.max()],
        'nearest_support': supports[0] if supports else low.min(),
        'nearest_resistance': resistances[0] if resistances else high.max(),
        'ma20': round(ma20, 4),
        'ma60': round(ma60, 4),
        'period_high': round(high.max(), 4),
        'period_low': round(low.min(), 4)
    }
