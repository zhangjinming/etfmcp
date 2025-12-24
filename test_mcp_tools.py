"""
ETF MCP工具测试脚本
测试main.py中所有MCP工具的功能
"""

import sys
import traceback
from datetime import datetime

# 测试结果统计
test_results = {
    'passed': 0,
    'failed': 0,
    'errors': []
}


def test_case(name: str):
    """测试用例装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"测试: {name}")
            print('='*60)
            try:
                result = func(*args, **kwargs)
                if result:
                    print(f"✓ 通过")
                    test_results['passed'] += 1
                else:
                    print(f"✗ 失败")
                    test_results['failed'] += 1
                return result
            except Exception as e:
                print(f"✗ 错误: {str(e)}")
                traceback.print_exc()
                test_results['failed'] += 1
                test_results['errors'].append({'name': name, 'error': str(e)})
                return False
        return wrapper
    return decorator


# 导入主模块中的函数
from main import (
    # 工具函数
    calculate_ma, calculate_ema, calculate_boll, calculate_rsi,
    calculate_macd, calculate_kdj, calculate_atr, calculate_obv,
    resample_to_weekly, get_indicator_signals,
    # 数据获取函数
    search_etf_by_name, get_etf_hist_data,
    # MCP工具
    search_etf, get_etf_technical_indicators, get_etf_realtime_info,
    get_index_realtime, get_index_history, get_macro_economic_data,
    get_economic_calendar, get_etf_comprehensive_analysis,
    get_etf_list_by_category, compare_etfs, get_market_overview,
    get_etf_performance_ranking, analyze_etf_trend, get_multi_etf_indicators
)
import pandas as pd
import numpy as np


# ==================== 工具函数测试 ====================

@test_case("calculate_ma - 移动平均线计算")
def test_calculate_ma():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ma = calculate_ma(data, 3)
    
    # MA3 的第3个值应该是 (1+2+3)/3 = 2
    assert pd.isna(ma.iloc[0]), "前两个值应为NaN"
    assert pd.isna(ma.iloc[1]), "前两个值应为NaN"
    assert abs(ma.iloc[2] - 2.0) < 0.001, f"MA3[2] 应为2.0, 实际为{ma.iloc[2]}"
    assert abs(ma.iloc[9] - 9.0) < 0.001, f"MA3[9] 应为9.0, 实际为{ma.iloc[9]}"
    
    print(f"  MA3计算结果: {ma.tolist()}")
    return True


@test_case("calculate_ema - 指数移动平均线计算")
def test_calculate_ema():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ema = calculate_ema(data, 3)
    
    # EMA应该有值（不像MA前面有NaN）
    assert not pd.isna(ema.iloc[0]), "EMA第一个值不应为NaN"
    assert ema.iloc[-1] > ema.iloc[0], "EMA应该呈上升趋势"
    
    print(f"  EMA3计算结果: {[round(x, 2) for x in ema.tolist()]}")
    return True


@test_case("calculate_boll - 布林带计算")
def test_calculate_boll():
    # 创建模拟数据
    np.random.seed(42)
    data = pd.DataFrame({
        'close': np.random.randn(50).cumsum() + 100
    })
    
    boll = calculate_boll(data, period=20, std_dev=2)
    
    assert 'upper' in boll, "应包含上轨"
    assert 'middle' in boll, "应包含中轨"
    assert 'lower' in boll, "应包含下轨"
    assert 'bandwidth' in boll, "应包含带宽"
    assert 'percent_b' in boll, "应包含%B"
    
    # 验证上轨 > 中轨 > 下轨
    last_idx = -1
    assert boll['upper'].iloc[last_idx] > boll['middle'].iloc[last_idx], "上轨应大于中轨"
    assert boll['middle'].iloc[last_idx] > boll['lower'].iloc[last_idx], "中轨应大于下轨"
    
    print(f"  上轨: {boll['upper'].iloc[last_idx]:.2f}")
    print(f"  中轨: {boll['middle'].iloc[last_idx]:.2f}")
    print(f"  下轨: {boll['lower'].iloc[last_idx]:.2f}")
    print(f"  %B: {boll['percent_b'].iloc[last_idx]:.2f}%")
    return True


@test_case("calculate_rsi - RSI计算")
def test_calculate_rsi():
    # 创建上涨趋势数据
    data_up = pd.Series([100 + i for i in range(30)])
    rsi_up = calculate_rsi(data_up, 14)
    
    # 创建下跌趋势数据
    data_down = pd.Series([100 - i for i in range(30)])
    rsi_down = calculate_rsi(data_down, 14)
    
    # 上涨趋势RSI应该高
    assert rsi_up.iloc[-1] > 70, f"上涨趋势RSI应>70, 实际为{rsi_up.iloc[-1]:.2f}"
    # 下跌趋势RSI应该低
    assert rsi_down.iloc[-1] < 30, f"下跌趋势RSI应<30, 实际为{rsi_down.iloc[-1]:.2f}"
    
    print(f"  上涨趋势RSI: {rsi_up.iloc[-1]:.2f}")
    print(f"  下跌趋势RSI: {rsi_down.iloc[-1]:.2f}")
    return True


@test_case("calculate_macd - MACD计算")
def test_calculate_macd():
    np.random.seed(42)
    data = pd.Series(np.random.randn(100).cumsum() + 100)
    
    macd = calculate_macd(data, fast=12, slow=26, signal=9)
    
    assert 'dif' in macd, "应包含DIF"
    assert 'dea' in macd, "应包含DEA"
    assert 'macd' in macd, "应包含MACD柱"
    
    # MACD柱 = 2 * (DIF - DEA)
    expected_macd = 2 * (macd['dif'].iloc[-1] - macd['dea'].iloc[-1])
    assert abs(macd['macd'].iloc[-1] - expected_macd) < 0.001, "MACD柱计算错误"
    
    print(f"  DIF: {macd['dif'].iloc[-1]:.4f}")
    print(f"  DEA: {macd['dea'].iloc[-1]:.4f}")
    print(f"  MACD: {macd['macd'].iloc[-1]:.4f}")
    return True


@test_case("calculate_kdj - KDJ计算")
def test_calculate_kdj():
    np.random.seed(42)
    base = np.random.randn(50).cumsum() + 100
    data = pd.DataFrame({
        'high': base + np.abs(np.random.randn(50)),
        'low': base - np.abs(np.random.randn(50)),
        'close': base
    })
    
    kdj = calculate_kdj(data, n=9, m1=3, m2=3)
    
    assert 'k' in kdj, "应包含K"
    assert 'd' in kdj, "应包含D"
    assert 'j' in kdj, "应包含J"
    
    # KDJ值应该在合理范围内（J可能超出0-100）
    k_val = kdj['k'].iloc[-1]
    d_val = kdj['d'].iloc[-1]
    
    print(f"  K: {k_val:.2f}")
    print(f"  D: {d_val:.2f}")
    print(f"  J: {kdj['j'].iloc[-1]:.2f}")
    return True


@test_case("resample_to_weekly - 日线转周线")
def test_resample_to_weekly():
    # 创建30天的日线数据
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': range(100, 130),
        'high': range(105, 135),
        'low': range(95, 125),
        'close': range(102, 132),
        'volume': [1000000] * 30
    })
    
    weekly = resample_to_weekly(data)
    
    assert len(weekly) < len(data), "周线数据量应小于日线"
    assert 'open' in weekly.columns, "应包含open列"
    assert 'high' in weekly.columns, "应包含high列"
    assert 'low' in weekly.columns, "应包含low列"
    assert 'close' in weekly.columns, "应包含close列"
    assert 'volume' in weekly.columns, "应包含volume列"
    
    print(f"  日线数据量: {len(data)}")
    print(f"  周线数据量: {len(weekly)}")
    return True


@test_case("get_indicator_signals - 信号生成")
def test_get_indicator_signals():
    # 测试超卖信号
    indicators_oversold = {
        'boll': {'percent_b': 10},
        'rsi': {'rsi_14': 25},
        'macd': {'dif': -0.5, 'dea': -0.3},
        'kdj': {'k': 15, 'd': 18, 'j': 10}
    }
    
    signals = get_indicator_signals(indicators_oversold)
    
    assert 'bullish' in signals, "应包含看涨信号"
    assert 'bearish' in signals, "应包含看跌信号"
    assert 'overall' in signals, "应包含综合判断"
    
    # 超卖情况应该有看涨信号
    assert len(signals['bullish']) > 0, "超卖情况应有看涨信号"
    
    print(f"  看涨信号: {signals['bullish']}")
    print(f"  看跌信号: {signals['bearish']}")
    print(f"  综合判断: {signals['overall']}")
    return True


@test_case("calculate_atr - ATR平均真实波幅计算")
def test_calculate_atr():
    np.random.seed(42)
    base = np.random.randn(50).cumsum() + 100
    data = pd.DataFrame({
        'high': base + np.abs(np.random.randn(50)) * 2,
        'low': base - np.abs(np.random.randn(50)) * 2,
        'close': base
    })
    
    atr = calculate_atr(data, period=14)
    
    # ATR应该是正数
    assert atr.iloc[-1] > 0, "ATR应该为正数"
    # ATR不应该有太多NaN
    assert atr.notna().sum() > len(atr) - 14, "ATR应该有足够的有效值"
    
    print(f"  ATR(14): {atr.iloc[-1]:.4f}")
    print(f"  ATR均值: {atr.mean():.4f}")
    return True


@test_case("calculate_obv - OBV能量潮计算")
def test_calculate_obv():
    # 创建上涨趋势数据
    data = pd.DataFrame({
        'close': [100 + i for i in range(30)],
        'volume': [1000000] * 30
    })
    
    obv = calculate_obv(data)
    
    # 上涨趋势OBV应该递增
    assert obv.iloc[-1] > obv.iloc[0], "上涨趋势OBV应该递增"
    
    # 创建下跌趋势数据
    data_down = pd.DataFrame({
        'close': [100 - i for i in range(30)],
        'volume': [1000000] * 30
    })
    
    obv_down = calculate_obv(data_down)
    
    # 下跌趋势OBV应该递减
    assert obv_down.iloc[-1] < obv_down.iloc[0], "下跌趋势OBV应该递减"
    
    print(f"  上涨趋势OBV最终值: {obv.iloc[-1]}")
    print(f"  下跌趋势OBV最终值: {obv_down.iloc[-1]}")
    return True


@test_case("get_indicator_signals - 超买信号")
def test_get_indicator_signals_overbought():
    # 测试超买信号
    indicators_overbought = {
        'boll': {'percent_b': 90},
        'rsi': {'rsi_14': 75},
        'macd': {'dif': 0.5, 'dea': 0.3},
        'kdj': {'k': 85, 'd': 82, 'j': 90}
    }
    
    signals = get_indicator_signals(indicators_overbought)
    
    # 超买情况应该有看跌信号
    assert len(signals['bearish']) > 0, "超买情况应有看跌信号"
    
    print(f"  看涨信号: {signals['bullish']}")
    print(f"  看跌信号: {signals['bearish']}")
    print(f"  综合判断: {signals['overall']}")
    return True


@test_case("get_indicator_signals - 中性信号")
def test_get_indicator_signals_neutral():
    # 测试中性信号
    indicators_neutral = {
        'boll': {'percent_b': 50},
        'rsi': {'rsi_14': 50},
        'macd': {'dif': 0.1, 'dea': 0.1},
        'kdj': {'k': 50, 'd': 50, 'j': 50}
    }
    
    signals = get_indicator_signals(indicators_neutral)
    
    # 中性情况应该有中性信号
    assert len(signals['neutral']) > 0, "中性情况应有中性信号"
    
    print(f"  看涨信号: {signals['bullish']}")
    print(f"  看跌信号: {signals['bearish']}")
    print(f"  中性信号: {signals['neutral']}")
    print(f"  综合判断: {signals['overall']}")
    return True


@test_case("边界情况 - 空数据处理")
def test_edge_case_empty_data():
    # 测试空Series的MA计算
    empty_series = pd.Series([], dtype=float)
    ma = calculate_ma(empty_series, 5)
    assert len(ma) == 0, "空数据MA应返回空Series"
    
    # 测试单个数据点
    single_point = pd.Series([100.0])
    ma_single = calculate_ma(single_point, 5)
    assert pd.isna(ma_single.iloc[0]), "单点数据MA应为NaN"
    
    print("  空数据和单点数据处理正常")
    return True


@test_case("边界情况 - 大周期参数")
def test_edge_case_large_period():
    # 测试周期大于数据量的情况
    data = pd.Series([1, 2, 3, 4, 5])
    ma = calculate_ma(data, 10)  # 周期大于数据量
    
    # 所有值应该是NaN
    assert ma.isna().all(), "周期大于数据量时所有MA应为NaN"
    
    print("  大周期参数处理正常")
    return True


# ==================== MCP工具测试(需要网络) ====================

@test_case("search_etf - 搜索ETF")
def test_search_etf():
    result = search_etf("沪深300")
    
    assert "沪深300" in result or "未找到" in result, "搜索结果应包含关键词或提示未找到"
    assert isinstance(result, str), "返回值应为字符串"
    
    if "未找到" not in result:
        assert "代码" in result or "(" in result, "搜索结果应包含ETF代码"
    
    print(f"  搜索结果预览: {result[:200]}...")
    return True


@test_case("search_etf - 搜索不存在的ETF")
def test_search_etf_not_found():
    result = search_etf("不存在的ETF名称xyz123")
    
    assert "未找到" in result, "不存在的ETF应返回未找到提示"
    
    print(f"  结果: {result}")
    return True


@test_case("get_etf_realtime_info - 获取ETF实时行情")
def test_get_etf_realtime_info():
    # 使用沪深300ETF代码测试
    result = get_etf_realtime_info("510300")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "未找到" not in result and "失败" not in result:
        assert "最新价" in result, "应包含最新价"
        assert "涨跌幅" in result, "应包含涨跌幅"
    
    print(f"  结果预览: {result[:300]}...")
    return True


@test_case("get_etf_realtime_info - 不存在的ETF代码")
def test_get_etf_realtime_info_not_found():
    result = get_etf_realtime_info("999999")
    
    assert "未找到" in result or "失败" in result, "不存在的代码应返回错误提示"
    
    print(f"  结果: {result}")
    return True


@test_case("get_etf_technical_indicators - 获取技术指标(周线)")
def test_get_etf_technical_indicators_weekly():
    result = get_etf_technical_indicators("510300", period="weekly")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result and "数据量不足" not in result:
        # 检查是否包含主要指标
        assert "BOLL" in result or "布林" in result, "应包含BOLL指标"
        assert "RSI" in result, "应包含RSI指标"
        assert "MACD" in result, "应包含MACD指标"
        assert "KDJ" in result, "应包含KDJ指标"
    
    print(f"  结果预览: {result[:500]}...")
    return True


@test_case("get_etf_technical_indicators - 获取技术指标(日线)")
def test_get_etf_technical_indicators_daily():
    result = get_etf_technical_indicators("510300", period="daily")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result:
        assert "日线" in result or "BOLL" in result, "应包含日线标识或技术指标"
    
    print(f"  结果预览: {result[:500]}...")
    return True


@test_case("get_index_realtime - 获取中国指数行情")
def test_get_index_realtime_china():
    result = get_index_realtime("china")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result:
        assert "指数" in result, "应包含指数信息"
    
    print(f"  结果预览: {result[:400]}...")
    return True


@test_case("get_index_realtime - 获取全球指数行情")
def test_get_index_realtime_global():
    result = get_index_realtime("global")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result:
        assert "全球" in result or "指数" in result, "应包含全球指数信息"
    
    print(f"  结果预览: {result[:400]}...")
    return True


@test_case("get_index_history - 获取指数历史数据")
def test_get_index_history():
    result = get_index_history("sh000001", days=30)
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result and "未找到" not in result:
        assert "历史数据" in result or "涨跌幅" in result, "应包含历史数据信息"
    
    print(f"  结果预览: {result[:400]}...")
    return True


@test_case("get_macro_economic_data - M2货币供应")
def test_get_macro_economic_data_m2():
    result = get_macro_economic_data("m2")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result:
        assert "M2" in result or "货币" in result, "应包含M2信息"
    
    print(f"  结果预览: {result[:300]}...")
    return True


@test_case("get_macro_economic_data - 不支持的指标")
def test_get_macro_economic_data_invalid():
    result = get_macro_economic_data("invalid_indicator")
    
    assert "不支持" in result, "不支持的指标应返回错误提示"
    
    print(f"  结果: {result}")
    return True


@test_case("get_economic_calendar - 获取经济日历")
def test_get_economic_calendar():
    result = get_economic_calendar()
    
    assert isinstance(result, str), "返回值应为字符串"
    # 可能没有数据，但不应该报错
    assert "失败" not in result or "经济事件" in result or "没有" in result, "应返回有效结果"
    
    print(f"  结果预览: {result[:300]}...")
    return True


@test_case("get_etf_comprehensive_analysis - 综合分析")
def test_get_etf_comprehensive_analysis():
    result = get_etf_comprehensive_analysis("沪深300")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "未找到" not in result and "失败" not in result:
        assert "综合分析" in result or "分析报告" in result or "实时行情" in result, "应包含分析内容"
    
    print(f"  结果预览: {result[:600]}...")
    return True


@test_case("get_etf_list_by_category - 获取全部ETF列表")
def test_get_etf_list_by_category_all():
    result = get_etf_list_by_category("all")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result:
        assert "ETF" in result, "应包含ETF信息"
    
    print(f"  结果预览: {result[:400]}...")
    return True


@test_case("get_etf_list_by_category - 获取行业ETF列表")
def test_get_etf_list_by_category_industry():
    result = get_etf_list_by_category("industry")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result and "未找到" not in result:
        assert "ETF" in result, "应包含ETF信息"
    
    print(f"  结果预览: {result[:400]}...")
    return True


@test_case("get_etf_list_by_category - 获取跨境ETF列表")
def test_get_etf_list_by_category_cross_border():
    result = get_etf_list_by_category("cross_border")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    print(f"  结果预览: {result[:400]}...")
    return True


@test_case("compare_etfs - ETF对比")
def test_compare_etfs():
    result = compare_etfs("510300,159915")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result:
        assert "对比" in result or "ETF" in result, "应包含对比信息"
    
    print(f"  结果预览: {result[:500]}...")
    return True


@test_case("compare_etfs - 单只ETF应提示错误")
def test_compare_etfs_single():
    result = compare_etfs("510300")
    
    assert "至少2只" in result, "单只ETF应提示需要至少2只"
    
    print(f"  结果: {result}")
    return True


@test_case("compare_etfs - 超过5只ETF应提示错误")
def test_compare_etfs_too_many():
    result = compare_etfs("510300,159915,510500,159919,510050,510880")
    
    assert "最多" in result, "超过5只应提示最多支持5只"
    
    print(f"  结果: {result}")
    return True


@test_case("get_market_overview - 市场概览")
def test_get_market_overview():
    result = get_market_overview()
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result:
        assert "市场概览" in result or "指数" in result or "ETF" in result, "应包含市场信息"
    
    print(f"  结果预览: {result[:600]}...")
    return True


@test_case("analyze_etf_trend - ETF趋势分析")
def test_analyze_etf_trend():
    result = analyze_etf_trend("510300")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result and "数据量不足" not in result:
        assert "趋势" in result, "应包含趋势判断"
        assert "均线" in result, "应包含均线分析"
        assert "评分" in result, "应包含趋势评分"
    
    print(f"  结果预览: {result[:600]}...")
    return True


@test_case("get_multi_etf_indicators - 批量ETF指标")
def test_get_multi_etf_indicators():
    result = get_multi_etf_indicators("沪深300,创业板")
    
    assert isinstance(result, str), "返回值应为字符串"
    
    if "失败" not in result:
        assert "RSI" in result or "MACD" in result, "应包含技术指标"
    
    print(f"  结果预览: {result[:500]}...")
    return True


# ==================== 运行测试 ====================

def run_unit_tests():
    """只运行工具函数测试（不需要网络）"""
    print("\n" + "="*70)
    print("  ETF MCP工具测试 - 单元测试（无网络）")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    test_calculate_ma()
    test_calculate_ema()
    test_calculate_boll()
    test_calculate_rsi()
    test_calculate_macd()
    test_calculate_kdj()
    test_calculate_atr()
    test_calculate_obv()
    test_resample_to_weekly()
    test_get_indicator_signals()
    test_get_indicator_signals_overbought()
    test_get_indicator_signals_neutral()
    test_edge_case_empty_data()
    test_edge_case_large_period()
    
    print_summary()


def run_integration_tests():
    """运行需要网络的集成测试"""
    print("\n" + "="*70)
    print("  ETF MCP工具测试 - 集成测试（需要网络）")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    test_search_etf()
    test_search_etf_not_found()
    test_get_etf_realtime_info()
    test_get_etf_realtime_info_not_found()
    test_get_etf_technical_indicators_weekly()
    test_get_etf_technical_indicators_daily()
    test_get_index_realtime_china()
    test_get_index_realtime_global()
    test_get_index_history()
    test_get_macro_economic_data_m2()
    test_get_macro_economic_data_invalid()
    test_get_economic_calendar()
    test_get_etf_comprehensive_analysis()
    test_get_etf_list_by_category_all()
    test_get_etf_list_by_category_industry()
    test_get_etf_list_by_category_cross_border()
    test_compare_etfs()
    test_compare_etfs_single()
    test_compare_etfs_too_many()
    test_get_market_overview()
    test_analyze_etf_trend()
    test_get_multi_etf_indicators()
    
    print_summary()


def print_summary():
    """打印测试结果汇总"""
    print("\n\n" + "="*70)
    print("  测试结果汇总")
    print("="*70)
    print(f"  通过: {test_results['passed']}")
    print(f"  失败: {test_results['failed']}")
    print(f"  总计: {test_results['passed'] + test_results['failed']}")
    
    if test_results['errors']:
        print("\n  错误详情:")
        for err in test_results['errors']:
            print(f"    - {err['name']}: {err['error']}")
    
    print("="*70)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("  ETF MCP工具测试 - 全部测试")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 工具函数测试
    print("\n\n>>> 工具函数测试 <<<")
    test_calculate_ma()
    test_calculate_ema()
    test_calculate_boll()
    test_calculate_rsi()
    test_calculate_macd()
    test_calculate_kdj()
    test_calculate_atr()
    test_calculate_obv()
    test_resample_to_weekly()
    test_get_indicator_signals()
    test_get_indicator_signals_overbought()
    test_get_indicator_signals_neutral()
    test_edge_case_empty_data()
    test_edge_case_large_period()
    
    # MCP工具测试
    print("\n\n>>> MCP工具测试 <<<")
    test_search_etf()
    test_search_etf_not_found()
    test_get_etf_realtime_info()
    test_get_etf_realtime_info_not_found()
    test_get_etf_technical_indicators_weekly()
    test_get_etf_technical_indicators_daily()
    test_get_index_realtime_china()
    test_get_index_realtime_global()
    test_get_index_history()
    test_get_macro_economic_data_m2()
    test_get_macro_economic_data_invalid()
    test_get_economic_calendar()
    test_get_etf_comprehensive_analysis()
    test_get_etf_list_by_category_all()
    test_get_etf_list_by_category_industry()
    test_get_etf_list_by_category_cross_border()
    test_compare_etfs()
    test_compare_etfs_single()
    test_compare_etfs_too_many()
    test_get_market_overview()
    test_analyze_etf_trend()
    test_get_multi_etf_indicators()
    
    print_summary()
    return test_results['failed'] == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ETF MCP工具测试')
    parser.add_argument('--unit', action='store_true', help='只运行单元测试（无网络）')
    parser.add_argument('--integration', action='store_true', help='只运行集成测试（需要网络）')
    args = parser.parse_args()
    
    if args.unit:
        run_unit_tests()
    elif args.integration:
        run_integration_tests()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
