"""
MCP 工具定义模块
定义所有暴露给 AI Agent 的 MCP 工具
"""

from datetime import datetime, timedelta
import pandas as pd
import akshare as ak

from cache import (
    get_cached_etf_spot,
    get_cached_index_spot_sina,
    get_cached_index_global_spot,
    get_cache,
    clear_cache as cache_clear,
    get_cache_stats as cache_stats,
    CACHE_TTL
)
from indicators import (
    calculate_ma,
    calculate_boll,
    calculate_rsi,
    calculate_macd,
    calculate_kdj,
    resample_to_weekly,
    format_indicator_summary,
    get_indicator_signals,
    calculate_period_score,
    analyze_historical_indicators,
    get_period_trend_judgment,
    analyze_weekly_trend,
    analyze_daily_buy_signals,
    analyze_daily_sell_signals,
    detect_macd_divergence,
    find_support_resistance
)
from data import (
    search_etf_by_name,
    get_etf_hist_data,
    get_index_hist_data
)


def register_tools(mcp):
    """注册所有 MCP 工具"""
    
    @mcp.tool()
    def search_etf(name: str) -> str:
        """
        根据ETF名称搜索ETF基金
        
        Args:
            name: ETF名称关键词，如"沪深300"、"纳斯达克"、"黄金"等
        
        Returns:
            匹配的ETF列表，包含代码、名称、最新价、涨跌幅
        """
        results = search_etf_by_name(name)
        if not results:
            return f"未找到包含'{name}'的ETF"
        
        if 'error' in results[0]:
            return f"搜索出错: {results[0]['error']}"
        
        output = f"搜索'{name}'找到以下ETF:\n\n"
        for i, etf in enumerate(results, 1):
            output += f"{i}. {etf['name']} ({etf['code']})\n"
            output += f"   最新价: {etf['latest_price']} | 涨跌幅: {etf['change_pct']}%\n"
        
        return output

    @mcp.tool()
    def get_etf_technical_indicators(code: str, period: str = "weekly", start_date: str = "", end_date: str = "") -> str:
        """
        获取ETF的技术指标分析（BOLL、RSI、MACD等）
        
        Args:
            code: ETF代码，如"159915"、"510300"
            period: 周期，"daily"日线或"weekly"周线，默认周线
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD，如"20240101"或"2024-01-01"
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD，如"20241225"或"2024-12-25"
        
        Returns:
            包含各项技术指标的详细分析报告
        """
        try:
            # 获取历史数据
            if start_date and end_date:
                df = get_etf_hist_data(code, start_date=start_date, end_date=end_date)
            else:
                df = get_etf_hist_data(code, days=365)
            
            if df.empty:
                return f"未能获取ETF {code} 的历史数据"
            
            # 保存日线数据用于计算涨跌幅
            daily_df = df.copy()
            
            # 根据周期转换数据
            if period == "weekly":
                df = resample_to_weekly(df)
            
            if len(df) < 30:
                return f"数据量不足，无法计算技术指标"
            
            # 获取ETF名称
            try:
                etf_info = get_cached_etf_spot()
                name_row = etf_info[etf_info['代码'] == code]
                etf_name = name_row['名称'].values[0] if not name_row.empty else code
            except:
                etf_name = code
            
            indicators = {}
            
            # 价格信息
            latest_price = df['close'].iloc[-1]
            
            # 周涨跌幅：当前价格与上周收盘价对比（周线数据）
            if len(df) > 1:
                week_ago_price = df['close'].iloc[-2]
                weekly_change_pct = round((latest_price - week_ago_price) / week_ago_price * 100, 2)
            else:
                weekly_change_pct = 0
            
            # 月涨跌幅：当前价格与上月最后一个交易日收盘价对比（与大智慧等软件一致）
            monthly_change_pct = 0
            if len(daily_df) > 0:
                # 确保日期列是datetime类型
                if 'date' in daily_df.columns:
                    daily_df['date'] = pd.to_datetime(daily_df['date'])
                    # 获取当前月份的第一天
                    current_date = daily_df['date'].iloc[-1]
                    first_day_of_month = current_date.replace(day=1)
                    
                    # 找到上月最后一个交易日的收盘价（本月1号之前的最后一个交易日）
                    last_month_df = daily_df[daily_df['date'] < first_day_of_month]
                    if len(last_month_df) > 0:
                        last_month_close_price = last_month_df['close'].iloc[-1]
                        monthly_change_pct = round((latest_price - last_month_close_price) / last_month_close_price * 100, 2)
            
            indicators['price_info'] = {
                'latest_price': round(latest_price, 4),
                'weekly_change_pct': weekly_change_pct,
                'monthly_change_pct': monthly_change_pct
            }
            
            # BOLL指标 (20日/周期)
            boll = calculate_boll(df, period=20, std_dev=2)
            boll_upper = boll['upper'].iloc[-1]
            boll_middle = boll['middle'].iloc[-1]
            boll_lower = boll['lower'].iloc[-1]
            boll_bandwidth = boll['bandwidth'].iloc[-1]
            boll_pb = boll['percent_b'].iloc[-1]
            
            indicators['boll'] = {
                'upper': round(boll_upper, 4),
                'middle': round(boll_middle, 4),
                'lower': round(boll_lower, 4),
                'bandwidth': round(boll_bandwidth, 2),
                'percent_b': round(boll_pb, 2)
            }
            
            # 判断BOLL信号
            if boll_pb < 20:
                indicators['boll']['signal'] = '接近下轨，可能超卖'
            elif boll_pb > 80:
                indicators['boll']['signal'] = '接近上轨，可能超买'
            elif boll_pb < 50:
                indicators['boll']['signal'] = '价格偏弱，在中轨下方'
            else:
                indicators['boll']['signal'] = '价格偏强，在中轨上方'
            
            # RSI指标
            rsi_6 = calculate_rsi(df['close'], 6).iloc[-1]
            rsi_12 = calculate_rsi(df['close'], 12).iloc[-1]
            rsi_14 = calculate_rsi(df['close'], 14).iloc[-1]
            
            indicators['rsi'] = {
                'rsi_6': round(rsi_6, 2),
                'rsi_12': round(rsi_12, 2),
                'rsi_14': round(rsi_14, 2)
            }
            
            if rsi_14 < 30:
                indicators['rsi']['signal'] = '超卖区域，可能反弹'
            elif rsi_14 > 70:
                indicators['rsi']['signal'] = '超买区域，可能回调'
            elif rsi_14 < 50:
                indicators['rsi']['signal'] = '偏弱势'
            else:
                indicators['rsi']['signal'] = '偏强势'
            
            # MACD指标 (12, 26, 9)
            macd = calculate_macd(df['close'], fast=12, slow=26, signal=9)
            dif = macd['dif'].iloc[-1]
            dea = macd['dea'].iloc[-1]
            macd_hist = macd['macd'].iloc[-1]
            
            # 检查MACD是否金叉/死叉（最近2个周期）
            prev_dif = macd['dif'].iloc[-2] if len(df) > 1 else dif
            prev_dea = macd['dea'].iloc[-2] if len(df) > 1 else dea
            
            indicators['macd'] = {
                'dif': round(dif, 4),
                'dea': round(dea, 4),
                'macd': round(macd_hist, 4)
            }
            
            # MACD信号判断
            if dif > dea and prev_dif <= prev_dea:
                indicators['macd']['signal'] = '刚形成金叉，短期看涨'
            elif dif < dea and prev_dif >= prev_dea:
                indicators['macd']['signal'] = '刚形成死叉，短期看跌'
            elif dif > dea and dif > 0:
                indicators['macd']['signal'] = '多头强势，DIF在零轴上方'
            elif dif < dea and dif < 0:
                indicators['macd']['signal'] = '空头强势，DIF在零轴下方'
            elif dif > dea:
                indicators['macd']['signal'] = '金叉状态，短期偏多'
            else:
                indicators['macd']['signal'] = '死叉状态，短期偏空'
            
            # 均线系统
            ma5 = calculate_ma(df['close'], 5).iloc[-1]
            ma10 = calculate_ma(df['close'], 10).iloc[-1]
            ma20 = calculate_ma(df['close'], 20).iloc[-1]
            ma60 = calculate_ma(df['close'], min(60, len(df)-1)).iloc[-1] if len(df) > 60 else None
            
            indicators['ma'] = {
                'ma5': round(ma5, 4),
                'ma10': round(ma10, 4),
                'ma20': round(ma20, 4),
                'ma60': round(ma60, 4) if ma60 else None
            }
            
            # 均线趋势判断
            if latest_price > ma5 > ma10 > ma20:
                indicators['ma']['trend'] = '多头排列，上升趋势'
            elif latest_price < ma5 < ma10 < ma20:
                indicators['ma']['trend'] = '空头排列，下降趋势'
            elif latest_price > ma5 and latest_price > ma10:
                indicators['ma']['trend'] = '短期偏多'
            elif latest_price < ma5 and latest_price < ma10:
                indicators['ma']['trend'] = '短期偏空'
            else:
                indicators['ma']['trend'] = '均线交织，震荡整理'
            
            # 成交量分析
            vol_ma5 = calculate_ma(df['volume'], 5).iloc[-1]
            current_vol = df['volume'].iloc[-1]
            indicators['volume'] = {
                'current': int(current_vol),
                'ma5': int(vol_ma5),
                'volume_ratio': round(current_vol / vol_ma5, 2) if vol_ma5 > 0 else 1
            }
            
            # ========== 综合趋势评分系统（百分制：0-100分）==========
            # 评分维度：均线(30分) + MACD(25分) + RSI(20分) + BOLL(15分) + 动量(10分)
            # 50分为中性，>50偏多，<50偏空
            
            trend_score = 50  # 基准分
            score_details = []
            
            # 1. 均线趋势评分（最高±15分，共30分权重）
            if latest_price > ma5 > ma10 > ma20:
                # 完美多头排列
                trend_score += 15
                score_details.append("均线多头排列(+15)")
            elif latest_price < ma5 < ma10 < ma20:
                # 完美空头排列
                trend_score -= 15
                score_details.append("均线空头排列(-15)")
            elif latest_price > ma5 and ma5 > ma10:
                # 短中期多头
                trend_score += 10
                score_details.append("短中期均线多头(+10)")
            elif latest_price < ma5 and ma5 < ma10:
                # 短中期空头
                trend_score -= 10
                score_details.append("短中期均线空头(-10)")
            elif latest_price > ma5:
                # 站上5日均线
                trend_score += 5
                score_details.append("价格在MA5上方(+5)")
            elif latest_price < ma5:
                # 跌破5日均线
                trend_score -= 5
                score_details.append("价格在MA5下方(-5)")
            
            # 2. MACD评分（最高±12.5分，共25分权重）
            if dif > dea and dif > 0:
                # 零轴上方金叉，强势
                trend_score += 12.5
                score_details.append("MACD零轴上金叉(+12.5)")
            elif dif > dea and dif < 0:
                # 零轴下方金叉，转强
                trend_score += 7.5
                score_details.append("MACD零轴下金叉(+7.5)")
            elif dif < dea and dif > 0:
                # 零轴上方死叉，转弱
                trend_score -= 7.5
                score_details.append("MACD零轴上死叉(-7.5)")
            elif dif < dea and dif < 0:
                # 零轴下方死叉，弱势
                trend_score -= 12.5
                score_details.append("MACD零轴下死叉(-12.5)")
            
            # 3. RSI评分（最高±10分，共20分权重）
            if rsi_14 >= 70:
                # 超买区，可能回调
                trend_score += 5  # 虽然超买但说明强势
                score_details.append(f"RSI超买{rsi_14:.1f}(+5)")
            elif rsi_14 >= 60:
                trend_score += 8
                score_details.append(f"RSI偏强{rsi_14:.1f}(+8)")
            elif rsi_14 >= 50:
                trend_score += 3
                score_details.append(f"RSI中性偏强{rsi_14:.1f}(+3)")
            elif rsi_14 >= 40:
                trend_score -= 3
                score_details.append(f"RSI中性偏弱{rsi_14:.1f}(-3)")
            elif rsi_14 >= 30:
                trend_score -= 8
                score_details.append(f"RSI偏弱{rsi_14:.1f}(-8)")
            else:
                # 超卖区
                trend_score -= 5  # 虽然超卖但可能反弹
                score_details.append(f"RSI超卖{rsi_14:.1f}(-5)")
            
            # 4. BOLL位置评分（最高±7.5分，共15分权重）
            if boll_pb >= 80:
                trend_score += 4  # 接近上轨，强势但注意回调
                score_details.append(f"BOLL接近上轨{boll_pb:.1f}%(+4)")
            elif boll_pb >= 60:
                trend_score += 6
                score_details.append(f"BOLL中上区域{boll_pb:.1f}%(+6)")
            elif boll_pb >= 40:
                trend_score += 0  # 中间区域，中性
                score_details.append(f"BOLL中间区域{boll_pb:.1f}%(0)")
            elif boll_pb >= 20:
                trend_score -= 6
                score_details.append(f"BOLL中下区域{boll_pb:.1f}%(-6)")
            else:
                trend_score -= 4  # 接近下轨，弱势但可能反弹
                score_details.append(f"BOLL接近下轨{boll_pb:.1f}%(-4)")
            
            # 5. 价格动量评分（最高±5分，共10分权重）
            if monthly_change_pct >= 10:
                trend_score += 5
                score_details.append(f"月涨幅{monthly_change_pct}%(+5)")
            elif monthly_change_pct >= 5:
                trend_score += 3
                score_details.append(f"月涨幅{monthly_change_pct}%(+3)")
            elif monthly_change_pct >= 0:
                trend_score += 1
                score_details.append(f"月涨幅{monthly_change_pct}%(+1)")
            elif monthly_change_pct >= -5:
                trend_score -= 1
                score_details.append(f"月跌幅{monthly_change_pct}%(-1)")
            elif monthly_change_pct >= -10:
                trend_score -= 3
                score_details.append(f"月跌幅{monthly_change_pct}%(-3)")
            else:
                trend_score -= 5
                score_details.append(f"月跌幅{monthly_change_pct}%(-5)")
            
            # 限制评分范围在0-100之间
            trend_score = max(0, min(100, trend_score))
            
            # 判断趋势类型（基于百分制评分）
            if trend_score >= 75:
                trend_type = "强势上涨"
                trend_desc = "多项指标共振向上，可持有或逢低加仓"
            elif trend_score >= 60:
                trend_type = "偏多震荡"
                trend_desc = "整体偏多但有波动，可轻仓参与"
            elif trend_score >= 45:
                trend_type = "横盘整理"
                trend_desc = "方向不明确，建议观望等待突破"
            elif trend_score >= 30:
                trend_type = "偏空震荡"
                trend_desc = "整体偏空，谨慎操作，控制仓位"
            else:
                trend_type = "弱势下跌"
                trend_desc = "多项指标显示弱势，建议观望或减仓"
            
            indicators['trend'] = {
                'type': trend_type,
                'score': round(trend_score, 1),
                'description': trend_desc,
                'details': score_details
            }
            
            # 生成信号汇总
            signals = get_indicator_signals(indicators)
            
            # 格式化输出
            period_name = "周线" if period == "weekly" else "日线"
            
            output = f"=== {etf_name}({code}) {period_name}技术指标分析 ===\n\n"
            
            # 价格信息
            output += f"【价格信息】\n"
            output += f"  最新价: {indicators['price_info']['latest_price']}\n"
            output += f"  周涨跌幅: {indicators['price_info']['weekly_change_pct']}%\n"
            output += f"  月涨跌幅: {indicators['price_info']['monthly_change_pct']}%\n\n"
            
            # BOLL
            output += f"【布林带 BOLL(20,2)】\n"
            output += f"  上轨: {indicators['boll']['upper']}\n"
            output += f"  中轨: {indicators['boll']['middle']}\n"
            output += f"  下轨: {indicators['boll']['lower']}\n"
            output += f"  带宽: {indicators['boll']['bandwidth']}%\n"
            output += f"  %B: {indicators['boll']['percent_b']}%\n"
            output += f"  信号: {indicators['boll']['signal']}\n\n"
            
            # RSI
            output += f"【RSI 相对强弱】\n"
            output += f"  RSI(6): {indicators['rsi']['rsi_6']}\n"
            output += f"  RSI(12): {indicators['rsi']['rsi_12']}\n"
            output += f"  RSI(14): {indicators['rsi']['rsi_14']}\n"
            output += f"  信号: {indicators['rsi']['signal']}\n\n"
            
            # MACD
            output += f"【MACD(12,26,9)】\n"
            output += f"  DIF: {indicators['macd']['dif']}\n"
            output += f"  DEA: {indicators['macd']['dea']}\n"
            output += f"  MACD柱: {indicators['macd']['macd']}\n"
            output += f"  信号: {indicators['macd']['signal']}\n\n"
            
            # 均线
            output += f"【均线系统】\n"
            output += f"  MA5: {indicators['ma']['ma5']}\n"
            output += f"  MA10: {indicators['ma']['ma10']}\n"
            output += f"  MA20: {indicators['ma']['ma20']}\n"
            if indicators['ma']['ma60']:
                output += f"  MA60: {indicators['ma']['ma60']}\n"
            output += f"  趋势: {indicators['ma']['trend']}\n\n"
            
            # 成交量
            output += f"【成交量分析】\n"
            output += f"  当前成交量: {indicators['volume']['current']}\n"
            output += f"  5日均量: {indicators['volume']['ma5']}\n"
            output += f"  量比: {indicators['volume']['volume_ratio']}\n\n"
            
            # 趋势判断
            output += "=" * 40 + "\n"
            output += f"【综合趋势评分】(百分制，50分为中性)\n"
            output += "=" * 40 + "\n"
            output += f"  趋势类型: {indicators['trend']['type']}\n"
            output += f"  综合评分: {indicators['trend']['score']}分\n"
            output += f"  操作建议: {indicators['trend']['description']}\n\n"
            output += "  评分明细:\n"
            for detail in indicators['trend']['details']:
                output += f"    • {detail}\n"
            output += "=" * 40 + "\n\n"
            
            # 信号汇总
            output += "=== 信号汇总 ===\n\n"
            
            if signals['bullish']:
                output += "【看涨信号】\n"
                for s in signals['bullish']:
                    output += f"  ✓ {s}\n"
                output += "\n"
            
            if signals['bearish']:
                output += "【看跌信号】\n"
                for s in signals['bearish']:
                    output += f"  ✗ {s}\n"
                output += "\n"
            
            if signals['neutral']:
                output += "【中性信号】\n"
                for s in signals['neutral']:
                    output += f"  - {s}\n"
                output += "\n"
            
            output += f"【综合判断】{signals['overall']}\n"
            
            return output
            
        except Exception as e:
            return f"获取技术指标失败: {str(e)}"

    @mcp.tool()
    def get_etf_realtime_info(code: str) -> str:
        """
        获取ETF实时行情信息
        
        Args:
            code: ETF代码，如"159915"、"510300"
        
        Returns:
            ETF的实时行情数据
        """
        try:
            etf_df = get_cached_etf_spot()
            etf_row = etf_df[etf_df['代码'] == code]
            
            if etf_row.empty:
                return f"未找到代码为 {code} 的ETF"
            
            row = etf_row.iloc[0]
            
            output = f"=== {row['名称']}({code}) 实时行情 ===\n\n"
            output += f"最新价: {row.get('最新价', 'N/A')}\n"
            output += f"涨跌额: {row.get('涨跌额', 'N/A')}\n"
            output += f"涨跌幅: {row.get('涨跌幅', 'N/A')}%\n"
            output += f"成交量: {row.get('成交量', 'N/A')}\n"
            output += f"成交额: {row.get('成交额', 'N/A')}\n"
            output += f"开盘价: {row.get('开盘价', 'N/A')}\n"
            output += f"最高价: {row.get('最高价', 'N/A')}\n"
            output += f"最低价: {row.get('最低价', 'N/A')}\n"
            output += f"昨收价: {row.get('昨收', 'N/A')}\n"
            output += f"换手率: {row.get('换手率', 'N/A')}%\n"
            
            return output
            
        except Exception as e:
            return f"获取实时行情失败: {str(e)}"

    @mcp.tool()
    def get_index_realtime(index_type: str = "china") -> str:
        """
        获取指数实时行情
        
        Args:
            index_type: 指数类型，"china"中国指数 或 "global"全球指数
        
        Returns:
            指数实时行情列表
        """
        try:
            if index_type == "global":
                df = get_cached_index_global_spot()
                output = "=== 全球主要指数实时行情 ===\n\n"
                
                # 选取主要指数
                important_indices = ['上证指数', '深证成指', '创业板指', '恒生指数', 
                                   '纳斯达克', '道琼斯', '标普500', '日经225', '德国DAX']
                
                for _, row in df.iterrows():
                    name = row.get('名称', '')
                    if any(idx in name for idx in important_indices) or len(output.split('\n')) < 25:
                        output += f"{row.get('名称', 'N/A')}: {row.get('最新价', 'N/A')} "
                        output += f"({row.get('涨跌幅', 'N/A')}%)\n"
            else:
                df = get_cached_index_spot_sina()
                output = "=== 中国主要指数实时行情 ===\n\n"
                
                # 主要指数代码
                important_codes = ['sh000001', 'sz399001', 'sz399006', 'sh000300', 
                                 'sh000016', 'sh000905', 'sz399673']
                
                for _, row in df.iterrows():
                    code = row.get('代码', '')
                    if code in important_codes:
                        output += f"{row.get('名称', 'N/A')}({code}): {row.get('最新价', 'N/A')} "
                        change_pct = row.get('涨跌幅', 0)
                        output += f"({change_pct}%)\n"
            
            return output
            
        except Exception as e:
            return f"获取指数行情失败: {str(e)}"

    @mcp.tool()
    def get_index_history(symbol: str, days: int = 60, start_date: str = "", end_date: str = "") -> str:
        """
        获取指数历史数据
        
        Args:
            symbol: 指数代码，如"sh000001"(上证指数)、"sz399001"(深证成指)
            days: 获取最近多少天的数据，默认60天（当start_date和end_date为空时使用）
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD，如"20240101"或"2024-01-01"
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD，如"20241225"或"2024-12-25"
        
        Returns:
            指数历史行情数据摘要
        """
        try:
            if start_date and end_date:
                df = get_index_hist_data(symbol, start_date=start_date, end_date=end_date)
            else:
                df = get_index_hist_data(symbol, days)
            
            if df.empty:
                return f"未找到指数 {symbol} 的历史数据"
            
            # 如果没有指定时间范围，取最近N天
            if not (start_date and end_date):
                df = df.tail(days)
            
            # 确定时间范围描述
            if start_date and end_date:
                period_desc = f"{start_date} 至 {end_date}"
            else:
                period_desc = f"最近{days}天"
            
            output = f"=== {symbol} {period_desc}历史数据 ===\n\n"
            
            # 统计信息
            latest = df.iloc[-1]
            first = df.iloc[0]
            
            output += f"期间涨跌幅: {round((latest['close'] - first['close']) / first['close'] * 100, 2)}%\n"
            output += f"最高价: {df['high'].max()} (日期: {df.loc[df['high'].idxmax(), 'date']})\n"
            output += f"最低价: {df['low'].min()} (日期: {df.loc[df['low'].idxmin(), 'date']})\n"
            output += f"平均成交量: {int(df['volume'].mean())}\n\n"
            
            output += "最近5个交易日:\n"
            for _, row in df.tail(5).iterrows():
                output += f"  {row['date']}: 开{row['open']} 高{row['high']} 低{row['low']} 收{row['close']}\n"
            
            return output
            
        except Exception as e:
            return f"获取指数历史数据失败: {str(e)}"

    @mcp.tool()
    def get_macro_economic_data(indicator: str) -> str:
        """
        获取宏观经济数据
        
        Args:
            indicator: 指标类型，可选值:
                - "m2": M2货币供应
                - "exports": 出口数据
                - "fx_reserves": 外汇储备
                - "enterprise_boom": 企业景气指数
                - "commodity_price": 大宗商品价格指数
                - "vegetable_basket": 菜篮子价格指数
        
        Returns:
            对应宏观经济指标数据
        """
        try:
            output = ""
            
            if indicator == "m2":
                df = ak.macro_china_m2_yearly()
                output = "=== M2货币供应年率 ===\n\n"
                for _, row in df.tail(12).iterrows():
                    output += f"{row.get('日期', row.get('date', 'N/A'))}: {row.get('今值', row.get('value', 'N/A'))}%\n"
                    
            elif indicator == "exports":
                df = ak.macro_china_exports_yoy()
                output = "=== 以美元计算出口年率 ===\n\n"
                for _, row in df.tail(12).iterrows():
                    output += f"{row.get('日期', row.get('date', 'N/A'))}: {row.get('今值', row.get('value', 'N/A'))}%\n"
                    
            elif indicator == "fx_reserves":
                df = ak.macro_china_fx_reserves_yearly()
                output = "=== 外汇储备(亿美元) ===\n\n"
                for _, row in df.tail(12).iterrows():
                    output += f"{row.get('日期', row.get('date', 'N/A'))}: {row.get('今值', row.get('value', 'N/A'))}\n"
                    
            elif indicator == "enterprise_boom":
                df = ak.macro_china_enterprise_boom_index()
                output = "=== 企业景气及企业家信心指数 ===\n\n"
                for _, row in df.tail(8).iterrows():
                    output += f"{row.get('季度', 'N/A')}: 景气指数{row.get('企业景气指数', 'N/A')} 信心指数{row.get('企业家信心指数', 'N/A')}\n"
                    
            elif indicator == "commodity_price":
                df = ak.macro_china_commodity_price_index()
                output = "=== 大宗商品价格指数 ===\n\n"
                for _, row in df.tail(12).iterrows():
                    output += f"{row.get('日期', 'N/A')}: {row.get('指数值', row.get('value', 'N/A'))}\n"
                    
            elif indicator == "vegetable_basket":
                df = ak.macro_china_vegetable_basket()
                output = "=== 菜篮子产品批发价格指数 ===\n\n"
                for _, row in df.tail(12).iterrows():
                    output += f"{row.get('日期', 'N/A')}: {row.get('指数值', row.get('value', 'N/A'))}\n"
            else:
                return f"不支持的指标类型: {indicator}。支持的类型: m2, exports, fx_reserves, enterprise_boom, commodity_price, vegetable_basket"
            
            return output
            
        except Exception as e:
            return f"获取宏观经济数据失败: {str(e)}"

    @mcp.tool()
    def get_economic_calendar(date: str = "") -> str:
        """
        获取全球宏观经济事件日历
        
        Args:
            date: 日期，格式YYYYMMDD，默认为今天
        
        Returns:
            当日重要经济事件列表
        """
        try:
            if not date:
                date = datetime.now().strftime('%Y%m%d')
            
            df = ak.news_economic_baidu(date=date)
            
            if df.empty:
                return f"{date} 没有重要经济事件"
            
            output = f"=== {date} 全球宏观经济事件 ===\n\n"
            
            for _, row in df.iterrows():
                output += f"【{row.get('时间', 'N/A')}】{row.get('地区', 'N/A')} - {row.get('事件', 'N/A')}\n"
                if row.get('前值'):
                    output += f"  前值: {row.get('前值', 'N/A')} | 预期: {row.get('预期', 'N/A')} | 公布: {row.get('公布', 'N/A')}\n"
                output += f"  重要性: {row.get('重要性', 'N/A')}\n\n"
            
            return output
            
        except Exception as e:
            return f"获取经济日历失败: {str(e)}"

    @mcp.tool()
    def get_etf_comprehensive_analysis(name: str, start_date: str = "", end_date: str = "") -> str:
        """
        根据ETF名称获取综合分析报告，包含技术指标、实时行情和相关宏观数据
        
        Args:
            name: ETF名称关键词，如"沪深300"、"纳斯达克"、"黄金"、"医药"等
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD，如"20240101"或"2024-01-01"
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD，如"20241225"或"2024-12-25"
        
        Returns:
            综合分析报告，整合多项指标供大模型判断
        """
        try:
            # 1. 搜索ETF
            etf_list = search_etf_by_name(name)
            
            if not etf_list or 'error' in etf_list[0]:
                return f"未找到包含'{name}'的ETF"
            
            # 取第一个匹配的ETF
            etf = etf_list[0]
            code = etf['code']
            etf_name = etf['name']
            
            output = f"{'='*50}\n"
            output += f"  {etf_name}({code}) 综合分析报告\n"
            output += f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            if start_date and end_date:
                output += f"  分析区间: {start_date} 至 {end_date}\n"
            output += f"{'='*50}\n\n"
            
            # 2. 实时行情
            try:
                etf_df = get_cached_etf_spot()
                etf_row = etf_df[etf_df['代码'] == code]
                
                if not etf_row.empty:
                    row = etf_row.iloc[0]
                    output += "【实时行情】\n"
                    output += f"  最新价: {row.get('最新价', 'N/A')}\n"
                    output += f"  涨跌幅: {row.get('涨跌幅', 'N/A')}%\n"
                    output += f"  成交额: {row.get('成交额', 'N/A')}\n"
                    output += f"  换手率: {row.get('换手率', 'N/A')}%\n\n"
            except:
                pass
            
            # 3. 周线技术指标
            try:
                if start_date and end_date:
                    df = get_etf_hist_data(code, start_date=start_date, end_date=end_date)
                else:
                    df = get_etf_hist_data(code, days=365)
                weekly_df = resample_to_weekly(df)
                
                if len(weekly_df) >= 30:
                    output += "【周线技术指标】\n"
                    
                    # BOLL
                    boll = calculate_boll(weekly_df)
                    pb = round(boll['percent_b'].iloc[-1], 2)
                    output += f"  BOLL %B: {pb}% "
                    if pb < 20:
                        output += "(接近下轨，超卖)\n"
                    elif pb > 80:
                        output += "(接近上轨，超买)\n"
                    else:
                        output += "(中间区域)\n"
                    
                    # RSI
                    rsi_14 = round(calculate_rsi(weekly_df['close'], 14).iloc[-1], 2)
                    output += f"  RSI(14): {rsi_14} "
                    if rsi_14 < 30:
                        output += "(超卖)\n"
                    elif rsi_14 > 70:
                        output += "(超买)\n"
                    else:
                        output += "(中性)\n"
                    
                    # MACD
                    macd = calculate_macd(weekly_df['close'])
                    dif = round(macd['dif'].iloc[-1], 4)
                    dea = round(macd['dea'].iloc[-1], 4)
                    output += f"  MACD DIF: {dif}, DEA: {dea} "
                    if dif > dea:
                        output += "(金叉/多头)\n"
                    else:
                        output += "(死叉/空头)\n"
                    
                    # 均线
                    ma5 = round(calculate_ma(weekly_df['close'], 5).iloc[-1], 4)
                    ma10 = round(calculate_ma(weekly_df['close'], 10).iloc[-1], 4)
                    ma20 = round(calculate_ma(weekly_df['close'], 20).iloc[-1], 4)
                    latest = weekly_df['close'].iloc[-1]
                    output += f"  均线: MA5={ma5}, MA10={ma10}, MA20={ma20}\n"
                    
                    if latest > ma5 > ma10 > ma20:
                        output += "  趋势: 多头排列\n"
                    elif latest < ma5 < ma10 < ma20:
                        output += "  趋势: 空头排列\n"
                    else:
                        output += "  趋势: 震荡整理\n"
                    
                    output += "\n"
            except Exception as e:
                output += f"  技术指标计算失败: {str(e)}\n\n"
            
            # 4. 历史表现
            try:
                if len(df) > 0:
                    output += "【历史表现】\n"
                    latest_price = df['close'].iloc[-1]
                    
                    # 近一周
                    if len(df) >= 5:
                        week_price = df['close'].iloc[-5]
                        week_change = round((latest_price - week_price) / week_price * 100, 2)
                        output += f"  近一周: {week_change}%\n"
                    
                    # 近一月
                    if len(df) >= 22:
                        month_price = df['close'].iloc[-22]
                        month_change = round((latest_price - month_price) / month_price * 100, 2)
                        output += f"  近一月: {month_change}%\n"
                    
                    # 近三月
                    if len(df) >= 66:
                        quarter_price = df['close'].iloc[-66]
                        quarter_change = round((latest_price - quarter_price) / quarter_price * 100, 2)
                        output += f"  近三月: {quarter_change}%\n"
                    
                    # 近一年
                    if len(df) >= 250:
                        year_price = df['close'].iloc[-250]
                        year_change = round((latest_price - year_price) / year_price * 100, 2)
                        output += f"  近一年: {year_change}%\n"
                    
                    output += "\n"
            except:
                pass
            
            # 5. 综合建议
            output += "【分析要点】\n"
            output += "  1. 以上技术指标基于周线数据，适合中期判断\n"
            output += "  2. RSI<30或BOLL%B<20可能是超卖信号\n"
            output += "  3. RSI>70或BOLL%B>80可能是超买信号\n"
            output += "  4. MACD金叉配合均线多头排列是较强的看涨信号\n"
            output += "  5. 建议结合宏观经济环境和行业基本面综合判断\n"
            
            return output
            
        except Exception as e:
            return f"综合分析失败: {str(e)}"

    @mcp.tool()
    def get_etf_list_by_category(category: str = "all") -> str:
        """
        获取ETF列表，按类别筛选
        
        Args:
            category: 类别，可选值:
                - "all": 全部ETF
                - "index": 指数ETF
                - "industry": 行业ETF
                - "commodity": 商品ETF
                - "bond": 债券ETF
                - "cross_border": 跨境ETF
        
        Returns:
            ETF列表
        """
        try:
            df = get_cached_etf_spot()
            
            if category != "all":
                category_keywords = {
                    "index": ["沪深300", "中证500", "上证50", "创业板", "科创"],
                    "industry": ["医药", "消费", "金融", "科技", "新能源", "半导体", "军工", "银行", "证券"],
                    "commodity": ["黄金", "白银", "原油", "有色", "能源"],
                    "bond": ["国债", "企债", "信用债", "可转债"],
                    "cross_border": ["纳斯达克", "标普", "恒生", "日经", "德国", "法国", "港股"]
                }
                
                keywords = category_keywords.get(category, [])
                if keywords:
                    mask = df['名称'].apply(lambda x: any(kw in str(x) for kw in keywords))
                    df = df[mask]
            
            if df.empty:
                return f"未找到{category}类别的ETF"
            
            output = f"=== {category.upper()} ETF列表 (共{len(df)}只) ===\n\n"
            
            # 按涨跌幅排序，显示前20只
            df_sorted = df.sort_values('涨跌幅', ascending=False)
            
            for _, row in df_sorted.head(20).iterrows():
                output += f"{row['名称']}({row['代码']}): {row.get('最新价', 'N/A')} ({row.get('涨跌幅', 'N/A')}%)\n"
            
            if len(df) > 20:
                output += f"\n... 共{len(df)}只，仅显示涨幅前20只\n"
            
            return output
            
        except Exception as e:
            return f"获取ETF列表失败: {str(e)}"

    @mcp.tool()
    def compare_etfs(codes: str) -> str:
        """
        比较多只ETF的表现
        
        Args:
            codes: ETF代码列表，用逗号分隔，如"510300,159915,510500"
        
        Returns:
            ETF对比分析报告
        """
        try:
            code_list = [c.strip() for c in codes.split(',')]
            
            if len(code_list) < 2:
                return "请提供至少2只ETF代码进行比较"
            
            if len(code_list) > 5:
                return "最多支持比较5只ETF"
            
            etf_df = get_cached_etf_spot()
            
            output = "=== ETF对比分析 ===\n\n"
            output += f"{'名称':<20} {'代码':<10} {'最新价':<10} {'涨跌幅':<10} {'换手率':<10}\n"
            output += "-" * 60 + "\n"
            
            comparison_data = []
            
            for code in code_list:
                row = etf_df[etf_df['代码'] == code]
                if not row.empty:
                    r = row.iloc[0]
                    name = r['名称'][:10]
                    output += f"{name:<20} {code:<10} {r.get('最新价', 'N/A'):<10} {r.get('涨跌幅', 'N/A')}%{'':<5} {r.get('换手率', 'N/A')}%\n"
                    
                    # 获取历史数据计算更多指标
                    try:
                        hist_df = get_etf_hist_data(code, days=250)
                        if len(hist_df) > 0:
                            latest = hist_df['close'].iloc[-1]
                            
                            week_ret = round((latest - hist_df['close'].iloc[-5]) / hist_df['close'].iloc[-5] * 100, 2) if len(hist_df) >= 5 else None
                            month_ret = round((latest - hist_df['close'].iloc[-22]) / hist_df['close'].iloc[-22] * 100, 2) if len(hist_df) >= 22 else None
                            
                            comparison_data.append({
                                'name': r['名称'],
                                'code': code,
                                'week_return': week_ret,
                                'month_return': month_ret
                            })
                    except:
                        pass
                else:
                    output += f"{'未找到':<20} {code:<10}\n"
            
            if comparison_data:
                output += "\n【历史收益对比】\n"
                output += f"{'名称':<15} {'周收益':<10} {'月收益':<10}\n"
                output += "-" * 35 + "\n"
                for d in comparison_data:
                    name = d['name'][:8]
                    output += f"{name:<15} {d['week_return']}%{'':<5} {d['month_return']}%\n"
            
            return output
            
        except Exception as e:
            return f"ETF对比失败: {str(e)}"

    @mcp.tool()
    def get_market_overview() -> str:
        """
        获取市场概览，包含主要指数、热门ETF和宏观数据摘要
        
        Returns:
            市场整体概览报告
        """
        try:
            output = f"{'='*50}\n"
            output += f"  市场概览 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            output += f"{'='*50}\n\n"
            
            # 1. 主要指数
            try:
                index_df = get_cached_index_spot_sina()
                important_codes = ['sh000001', 'sz399001', 'sz399006', 'sh000300']
                
                output += "【主要指数】\n"
                for _, row in index_df.iterrows():
                    if row.get('代码', '') in important_codes:
                        output += f"  {row['名称']}: {row['最新价']} ({row.get('涨跌幅', 0)}%)\n"
                output += "\n"
            except:
                pass
            
            # 2. 热门ETF
            try:
                etf_df = get_cached_etf_spot()
                etf_sorted = etf_df.sort_values('成交额', ascending=False)
                
                output += "【成交额前10 ETF】\n"
                for _, row in etf_sorted.head(10).iterrows():
                    output += f"  {row['名称']}: {row.get('最新价', 'N/A')} ({row.get('涨跌幅', 'N/A')}%)\n"
                output += "\n"
            except:
                pass
            
            # 3. 涨幅榜
            try:
                etf_up = etf_df.sort_values('涨跌幅', ascending=False)
                output += "【涨幅前5 ETF】\n"
                for _, row in etf_up.head(5).iterrows():
                    output += f"  {row['名称']}: +{row.get('涨跌幅', 'N/A')}%\n"
                output += "\n"
                
                # 跌幅榜
                etf_down = etf_df.sort_values('涨跌幅', ascending=True)
                output += "【跌幅前5 ETF】\n"
                for _, row in etf_down.head(5).iterrows():
                    output += f"  {row['名称']}: {row.get('涨跌幅', 'N/A')}%\n"
                output += "\n"
            except:
                pass
            
            return output
            
        except Exception as e:
            return f"获取市场概览失败: {str(e)}"

    @mcp.tool()
    def get_etf_performance_ranking(period: str = "week", top_n: int = 10) -> str:
        """
        获取ETF涨跌幅排行榜
        
        Args:
            period: 排行周期，"day"当日、"week"近一周、"month"近一月
            top_n: 显示前N只，默认10
        
        Returns:
            ETF涨跌幅排行榜
        """
        try:
            etf_df = get_cached_etf_spot()
            
            if period == "day":
                # 当日涨跌幅
                df_sorted_up = etf_df.sort_values('涨跌幅', ascending=False)
                df_sorted_down = etf_df.sort_values('涨跌幅', ascending=True)
                period_name = "当日"
            else:
                # 需要计算历史涨跌幅
                results = []
                
                for _, row in etf_df.iterrows():
                    try:
                        code = row['代码']
                        hist_df = get_etf_hist_data(code, days=30)
                        
                        if len(hist_df) > 0:
                            latest = hist_df['close'].iloc[-1]
                            
                            if period == "week" and len(hist_df) >= 5:
                                base_price = hist_df['close'].iloc[-5]
                                change = (latest - base_price) / base_price * 100
                            elif period == "month" and len(hist_df) >= 22:
                                base_price = hist_df['close'].iloc[-22]
                                change = (latest - base_price) / base_price * 100
                            else:
                                continue
                            
                            results.append({
                                'code': code,
                                'name': row['名称'],
                                'change': round(change, 2)
                            })
                    except:
                        continue
                    
                    # 限制查询数量避免超时
                    if len(results) >= 50:
                        break
                
                if not results:
                    return f"无法获取{period}周期的排行数据"
                
                results_df = pd.DataFrame(results)
                df_sorted_up = results_df.sort_values('change', ascending=False)
                df_sorted_down = results_df.sort_values('change', ascending=True)
                period_name = "近一周" if period == "week" else "近一月"
            
            output = f"=== ETF {period_name}涨跌幅排行 ===\n\n"
            
            output += f"【涨幅前{top_n}】\n"
            if period == "day":
                for i, (_, row) in enumerate(df_sorted_up.head(top_n).iterrows(), 1):
                    output += f"  {i}. {row['名称']}: +{row.get('涨跌幅', 'N/A')}%\n"
            else:
                for i, (_, row) in enumerate(df_sorted_up.head(top_n).iterrows(), 1):
                    output += f"  {i}. {row['name']}: +{row['change']}%\n"
            
            output += f"\n【跌幅前{top_n}】\n"
            if period == "day":
                for i, (_, row) in enumerate(df_sorted_down.head(top_n).iterrows(), 1):
                    output += f"  {i}. {row['名称']}: {row.get('涨跌幅', 'N/A')}%\n"
            else:
                for i, (_, row) in enumerate(df_sorted_down.head(top_n).iterrows(), 1):
                    output += f"  {i}. {row['name']}: {row['change']}%\n"
            
            return output
            
        except Exception as e:
            return f"获取排行榜失败: {str(e)}"

    @mcp.tool()
    def analyze_etf_trend(code: str, start_date: str = "", end_date: str = "") -> str:
        """
        分析ETF的趋势状态，包含当前指标、近半年和近一年的技术指标历史统计和综合评分
        
        Args:
            code: ETF代码，如"510300"
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD，如"20240101"或"2024-01-01"
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD，如"20241225"或"2024-12-25"
        
        Returns:
            趋势分析报告，包含多周期技术指标统计、趋势判断和综合评分
        """
        try:
            # 获取历史数据（2年日线数据）
            if start_date and end_date:
                df = get_etf_hist_data(code, start_date=start_date, end_date=end_date)
            else:
                df = get_etf_hist_data(code, days=730)
            
            if df.empty or len(df) < 60:
                return f"数据量不足，无法分析趋势"
            
            # 转换为周线数据
            weekly_df = resample_to_weekly(df)
            
            if len(weekly_df) < 30:
                return f"周线数据量不足，无法分析趋势"
            
            # 获取ETF名称
            try:
                etf_info = get_cached_etf_spot()
                name_row = etf_info[etf_info['代码'] == code]
                etf_name = name_row['名称'].values[0] if not name_row.empty else code
            except:
                etf_name = code
            
            # ========== 1. 当前技术指标 ==========
            current_score_data = calculate_period_score(weekly_df)
            if not current_score_data:
                return "计算当前指标失败"
            
            latest_price = weekly_df['close'].iloc[-1]
            
            # ========== 2. 历史周期分析 ==========
            # 近13周（约3个月/一季度）
            stats_3m = analyze_historical_indicators(weekly_df, 13)
            score_3m, judgments_3m = get_period_trend_judgment(stats_3m)
            
            # 近26周（约半年）
            stats_6m = analyze_historical_indicators(weekly_df, 26)
            score_6m, judgments_6m = get_period_trend_judgment(stats_6m)
            
            # 近52周（约一年）
            stats_1y = analyze_historical_indicators(weekly_df, 52)
            score_1y, judgments_1y = get_period_trend_judgment(stats_1y)
            
            # ========== 3. 综合评分 ==========
            # 当前指标权重40%，近3月20%，近半年20%，近一年20%
            current_score = current_score_data['score']
            comprehensive_score = int(current_score * 0.4 + score_3m * 0.2 + score_6m * 0.2 + score_1y * 0.2)
            
            # 趋势判断
            if comprehensive_score >= 40:
                trend = "强势上涨趋势"
                suggestion = "可考虑持有或逢低加仓"
            elif comprehensive_score >= 15:
                trend = "偏多震荡趋势"
                suggestion = "可考虑轻仓参与，注意回调风险"
            elif comprehensive_score >= -15:
                trend = "横盘整理"
                suggestion = "建议观望，等待方向明确"
            elif comprehensive_score >= -40:
                trend = "偏空震荡趋势"
                suggestion = "建议减仓或观望，谨慎操作"
            else:
                trend = "弱势下跌趋势"
                suggestion = "建议回避或空仓等待企稳"
            
            # ========== 生成报告 ==========
            output = f"{'='*60}\n"
            output += f"  {etf_name}({code}) 多周期技术指标分析报告\n"
            output += f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            if start_date and end_date:
                output += f"  分析区间: {start_date} 至 {end_date}\n"
            output += f"{'='*60}\n\n"
            
            output += f"【当前价格】{latest_price:.4f}\n\n"
            
            # 当前指标
            output += "=" * 40 + "\n"
            output += "【当前周线技术指标】\n"
            output += "=" * 40 + "\n"
            output += f"  BOLL %B: {current_score_data['percent_b']:.1f}%\n"
            output += f"  RSI(14): {current_score_data['rsi']:.1f}\n"
            output += f"  MACD DIF: {current_score_data['dif']:.4f}, DEA: {current_score_data['dea']:.4f}\n"
            output += f"  MA5周: {current_score_data['ma5']:.4f} {'↑' if latest_price > current_score_data['ma5'] else '↓'}\n"
            output += f"  MA10周: {current_score_data['ma10']:.4f} {'↑' if latest_price > current_score_data['ma10'] else '↓'}\n"
            output += f"  MA20周: {current_score_data['ma20']:.4f} {'↑' if latest_price > current_score_data['ma20'] else '↓'}\n"
            output += f"  量比: {current_score_data['volume_ratio']:.2f}\n"
            output += f"  当前评分: {current_score}分\n"
            output += f"  信号: {', '.join(current_score_data['details'])}\n\n"
            
            # 近3个月统计
            output += "=" * 40 + "\n"
            output += f"【近3个月({stats_3m['weeks']}周)技术指标统计】\n"
            output += "=" * 40 + "\n"
            output += f"  区间涨跌幅: {stats_3m['total_change']}%\n"
            output += f"  最大回撤: {stats_3m['max_drawdown']}%\n"
            output += f"  最大涨幅: {stats_3m['max_rally']}%\n"
            output += f"  上涨周数/下跌周数: {stats_3m['up_weeks']}/{stats_3m['down_weeks']}\n"
            output += f"  RSI范围: {stats_3m['rsi_min']} ~ {stats_3m['rsi_max']} (均值{stats_3m['rsi_avg']})\n"
            output += f"  RSI超卖/超买次数: {stats_3m['rsi_oversold_count']}/{stats_3m['rsi_overbought_count']}\n"
            output += f"  BOLL%B范围: {stats_3m['pb_min']}% ~ {stats_3m['pb_max']}% (均值{stats_3m['pb_avg']}%)\n"
            output += f"  BOLL触下轨/上轨次数: {stats_3m['pb_near_lower']}/{stats_3m['pb_near_upper']}\n"
            output += f"  MACD金叉/死叉次数: {stats_3m['macd_cross_up']}/{stats_3m['macd_cross_down']}\n"
            output += f"  周期评分: {score_3m}分\n"
            output += f"  特征: {', '.join(judgments_3m)}\n\n"
            
            # 近半年统计
            output += "=" * 40 + "\n"
            output += f"【近半年({stats_6m['weeks']}周)技术指标统计】\n"
            output += "=" * 40 + "\n"
            output += f"  区间涨跌幅: {stats_6m['total_change']}%\n"
            output += f"  最大回撤: {stats_6m['max_drawdown']}%\n"
            output += f"  最大涨幅: {stats_6m['max_rally']}%\n"
            output += f"  上涨周数/下跌周数: {stats_6m['up_weeks']}/{stats_6m['down_weeks']}\n"
            output += f"  RSI范围: {stats_6m['rsi_min']} ~ {stats_6m['rsi_max']} (均值{stats_6m['rsi_avg']})\n"
            output += f"  RSI超卖/超买次数: {stats_6m['rsi_oversold_count']}/{stats_6m['rsi_overbought_count']}\n"
            output += f"  BOLL%B范围: {stats_6m['pb_min']}% ~ {stats_6m['pb_max']}% (均值{stats_6m['pb_avg']}%)\n"
            output += f"  BOLL触下轨/上轨次数: {stats_6m['pb_near_lower']}/{stats_6m['pb_near_upper']}\n"
            output += f"  MACD金叉/死叉次数: {stats_6m['macd_cross_up']}/{stats_6m['macd_cross_down']}\n"
            output += f"  周期评分: {score_6m}分\n"
            output += f"  特征: {', '.join(judgments_6m)}\n\n"
            
            # 近一年统计
            output += "=" * 40 + "\n"
            output += f"【近一年({stats_1y['weeks']}周)技术指标统计】\n"
            output += "=" * 40 + "\n"
            output += f"  区间涨跌幅: {stats_1y['total_change']}%\n"
            output += f"  最大回撤: {stats_1y['max_drawdown']}%\n"
            output += f"  最大涨幅: {stats_1y['max_rally']}%\n"
            output += f"  上涨周数/下跌周数: {stats_1y['up_weeks']}/{stats_1y['down_weeks']}\n"
            output += f"  RSI范围: {stats_1y['rsi_min']} ~ {stats_1y['rsi_max']} (均值{stats_1y['rsi_avg']})\n"
            output += f"  RSI超卖/超买次数: {stats_1y['rsi_oversold_count']}/{stats_1y['rsi_overbought_count']}\n"
            output += f"  BOLL%B范围: {stats_1y['pb_min']}% ~ {stats_1y['pb_max']}% (均值{stats_1y['pb_avg']}%)\n"
            output += f"  BOLL触下轨/上轨次数: {stats_1y['pb_near_lower']}/{stats_1y['pb_near_upper']}\n"
            output += f"  MACD金叉/死叉次数: {stats_1y['macd_cross_up']}/{stats_1y['macd_cross_down']}\n"
            output += f"  周期评分: {score_1y}分\n"
            output += f"  特征: {', '.join(judgments_1y)}\n\n"
            
            # 综合评分
            output += "=" * 60 + "\n"
            output += "【综合评分与趋势判断】\n"
            output += "=" * 60 + "\n"
            output += f"  当前指标评分(权重40%): {current_score}分\n"
            output += f"  近3个月评分(权重20%): {score_3m}分\n"
            output += f"  近半年评分(权重20%): {score_6m}分\n"
            output += f"  近一年评分(权重20%): {score_1y}分\n"
            output += f"  ─────────────────────\n"
            output += f"  【综合评分】{comprehensive_score}分\n"
            output += f"  【趋势判断】{trend}\n"
            output += f"  【操作建议】{suggestion}\n\n"
            
            # 风险提示
            output += "【风险提示】\n"
            output += "  • 以上分析基于周线数据，适合中长期投资参考\n"
            output += "  • 历史表现不代表未来收益，投资有风险\n"
            output += "  • 建议结合基本面和宏观环境综合判断\n"
            
            return output
            
        except Exception as e:
            return f"趋势分析失败: {str(e)}"

    @mcp.tool()
    def get_multi_etf_indicators(names: str) -> str:
        """
        批量获取多只ETF的关键技术指标摘要
        
        Args:
            names: ETF名称列表，用逗号分隔，如"沪深300,创业板,纳斯达克"
        
        Returns:
            多只ETF的技术指标对比表
        """
        try:
            name_list = [n.strip() for n in names.split(',')]
            
            if len(name_list) > 10:
                return "最多支持同时查询10只ETF"
            
            results = []
            
            for name in name_list:
                etf_list = search_etf_by_name(name)
                if not etf_list or 'error' in etf_list[0]:
                    continue
                
                etf = etf_list[0]
                code = etf['code']
                
                try:
                    df = get_etf_hist_data(code, days=120)
                    if len(df) < 30:
                        continue
                    
                    weekly_df = resample_to_weekly(df)
                    
                    # 计算指标
                    rsi_14 = calculate_rsi(weekly_df['close'], 14).iloc[-1]
                    macd = calculate_macd(weekly_df['close'])
                    boll = calculate_boll(weekly_df)
                    
                    # 近期涨跌幅
                    latest = df['close'].iloc[-1]
                    week_change = round((latest - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100, 2) if len(df) >= 5 else None
                    month_change = round((latest - df['close'].iloc[-22]) / df['close'].iloc[-22] * 100, 2) if len(df) >= 22 else None
                    
                    results.append({
                        'name': etf['name'][:12],
                        'code': code,
                        'price': latest,
                        'week_change': week_change,
                        'month_change': month_change,
                        'rsi': round(rsi_14, 1),
                        'macd_signal': '多' if macd['dif'].iloc[-1] > macd['dea'].iloc[-1] else '空',
                        'boll_pb': round(boll['percent_b'].iloc[-1], 1)
                    })
                except:
                    continue
            
            if not results:
                return "未能获取任何ETF数据"
            
            output = "=== 多ETF技术指标对比 ===\n\n"
            output += f"{'名称':<14} {'代码':<8} {'价格':<8} {'周涨跌':<8} {'月涨跌':<8} {'RSI':<6} {'MACD':<6} {'BOLL%B':<8}\n"
            output += "-" * 80 + "\n"
            
            for r in results:
                output += f"{r['name']:<14} {r['code']:<8} {r['price']:<8.3f} "
                output += f"{r['week_change']}%{'':<3} " if r['week_change'] else "N/A      "
                output += f"{r['month_change']}%{'':<3} " if r['month_change'] else "N/A      "
                output += f"{r['rsi']:<6} {r['macd_signal']:<6} {r['boll_pb']}%\n"
            
            output += "\n【指标说明】\n"
            output += "  RSI: <30超卖, >70超买\n"
            output += "  MACD: 多=金叉, 空=死叉\n"
            output += "  BOLL%B: <20接近下轨, >80接近上轨\n"
            
            return output
            
        except Exception as e:
            return f"批量查询失败: {str(e)}"

    @mcp.tool()
    def get_etf_history(code: str, days: int = 60, start_date: str = "", end_date: str = "") -> str:
        """
        获取ETF历史行情数据
        
        Args:
            code: ETF代码，如"510300"、"159915"
            days: 获取最近多少天的数据，默认60天（当start_date和end_date为空时使用）
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD，如"20240101"或"2024-01-01"
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD，如"20241225"或"2024-12-25"
        
        Returns:
            ETF历史行情数据摘要
        """
        try:
            # 获取ETF名称
            try:
                etf_info = get_cached_etf_spot()
                name_row = etf_info[etf_info['代码'] == code]
                etf_name = name_row['名称'].values[0] if not name_row.empty else code
            except:
                etf_name = code
            
            # 获取历史数据
            if start_date and end_date:
                df = get_etf_hist_data(code, start_date=start_date, end_date=end_date)
                period_desc = f"{start_date} 至 {end_date}"
            else:
                df = get_etf_hist_data(code, days=days)
                period_desc = f"最近{days}天"
            
            if df.empty:
                return f"未找到ETF {code} 的历史数据"
            
            output = f"=== {etf_name}({code}) {period_desc}历史数据 ===\n\n"
            
            # 统计信息
            latest = df.iloc[-1]
            first = df.iloc[0]
            
            output += f"数据条数: {len(df)}条\n"
            output += f"起始日期: {df['date'].iloc[0].strftime('%Y-%m-%d')}\n"
            output += f"结束日期: {df['date'].iloc[-1].strftime('%Y-%m-%d')}\n"
            output += f"期间涨跌幅: {round((latest['close'] - first['close']) / first['close'] * 100, 2)}%\n"
            output += f"最高价: {df['high'].max():.4f} (日期: {df.loc[df['high'].idxmax(), 'date'].strftime('%Y-%m-%d')})\n"
            output += f"最低价: {df['low'].min():.4f} (日期: {df.loc[df['low'].idxmin(), 'date'].strftime('%Y-%m-%d')})\n"
            output += f"平均成交量: {int(df['volume'].mean())}\n\n"
            
            output += "最近5个交易日:\n"
            for _, row in df.tail(5).iterrows():
                output += f"  {row['date'].strftime('%Y-%m-%d')}: 开{row['open']:.4f} 高{row['high']:.4f} 低{row['low']:.4f} 收{row['close']:.4f}\n"
            
            return output
            
        except Exception as e:
            return f"获取ETF历史数据失败: {str(e)}"

    @mcp.tool()
    def clear_data_cache() -> str:
        """
        清除数据缓存，强制下次请求重新获取最新数据
        
        Returns:
            缓存清除结果
        """
        stats = cache_clear()
        return f"缓存已清除。清除前缓存项数: {stats['cache_size']}"

    @mcp.tool()
    def get_data_cache_stats() -> str:
        """
        获取当前缓存状态信息
        
        Returns:
            缓存统计信息
        """
        stats = cache_stats()
        output = "=== 缓存状态 ===\n\n"
        output += f"缓存项数: {stats['cache_size']}\n"
        output += f"缓存键列表:\n"
        for key in stats['keys']:
            output += f"  - {key}\n"
        output += f"\n缓存过期时间配置:\n"
        for k, v in CACHE_TTL.items():
            output += f"  {k}: {v}秒\n"
        return output

    @mcp.tool()
    def get_csv_data_info() -> str:
        """
        获取本地CSV数据文件信息
        
        Returns:
            CSV数据文件列表及统计信息
        """
        from data import DATA_DIR
        
        output = "=== 本地CSV数据文件 ===\n\n"
        output += f"数据目录: {DATA_DIR}\n\n"
        
        csv_files = list(DATA_DIR.glob("*.csv"))
        
        if not csv_files:
            output += "暂无本地数据文件\n"
            return output
        
        output += f"共 {len(csv_files)} 个数据文件:\n\n"
        
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    min_date = df['date'].min().strftime('%Y-%m-%d')
                    max_date = df['date'].max().strftime('%Y-%m-%d')
                    output += f"  {csv_file.name}:\n"
                    output += f"    数据条数: {len(df)}\n"
                    output += f"    时间范围: {min_date} ~ {max_date}\n"
                else:
                    output += f"  {csv_file.name}: {len(df)} 条数据\n"
            except Exception as e:
                output += f"  {csv_file.name}: 读取失败 ({e})\n"
        
        return output

    @mcp.tool()
    def clear_csv_data(code: str = "") -> str:
        """
        清除本地CSV数据文件
        
        Args:
            code: 指定要清除的ETF/指数代码，为空则清除所有数据文件
        
        Returns:
            清除结果
        """
        from data import DATA_DIR
        
        if code:
            # 清除指定代码的数据
            deleted = []
            for pattern in [f"etf_{code}.csv", f"index_{code}.csv"]:
                csv_path = DATA_DIR / pattern
                if csv_path.exists():
                    csv_path.unlink()
                    deleted.append(pattern)
            
            if deleted:
                return f"已删除数据文件: {', '.join(deleted)}"
            else:
                return f"未找到代码 {code} 的数据文件"
        else:
            # 清除所有数据
            csv_files = list(DATA_DIR.glob("*.csv"))
            count = len(csv_files)
            
            for csv_file in csv_files:
                csv_file.unlink()
            
            return f"已清除 {count} 个CSV数据文件"

    @mcp.tool()
    def download_etf_data(code: str, start_date: str = "", end_date: str = "") -> str:
        """
        下载ETF历史数据到本地CSV文件
        
        Args:
            code: ETF代码，如"510300"
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD，默认为2年前
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD，默认为今天
        
        Returns:
            下载结果
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            
            # 调用数据获取函数，会自动保存到CSV
            df = get_etf_hist_data(code, start_date=start_date, end_date=end_date)
            
            if df.empty:
                return f"未能获取ETF {code} 的数据"
            
            from data import DATA_DIR, _get_csv_path
            csv_path = _get_csv_path('etf', code)
            
            return f"ETF {code} 数据下载完成:\n  数据条数: {len(df)}\n  时间范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}\n  保存路径: {csv_path}"
            
        except Exception as e:
            return f"下载ETF数据失败: {str(e)}"

    @mcp.tool()
    def download_index_data(symbol: str) -> str:
        """
        下载指数历史数据到本地CSV文件
        
        Args:
            symbol: 指数代码，如"sh000001"(上证指数)
        
        Returns:
            下载结果
        """
        try:
            # 调用数据获取函数，会自动保存到CSV
            df = get_index_hist_data(symbol, days=730)
            
            if df.empty:
                return f"未能获取指数 {symbol} 的数据"
            
            from data import DATA_DIR, _get_csv_path
            csv_path = _get_csv_path('index', symbol)
            
            return f"指数 {symbol} 数据下载完成:\n  数据条数: {len(df)}\n  时间范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}\n  保存路径: {csv_path}"
            
        except Exception as e:
            return f"下载指数数据失败: {str(e)}"

    # ========== 波段交易策略工具 ==========

    @mcp.tool()
    def get_swing_trade_analysis(name: str) -> str:
        """
        波段交易综合分析（核心工具）
        基于"周线定方向，日线找买卖点"的策略框架
        
        策略说明：
        1. 周线判断中期趋势（1-6个月），使用MA20/MA60判断多空
        2. 日线寻找具体买卖点，使用MACD背离、KDJ超卖金叉等信号
        3. 只在周线多头趋势中寻找日线回调买入机会
        
        Args:
            name: ETF名称关键词，如"沪深300"、"创业板"、"纳斯达克"等
        
        Returns:
            完整的波段交易分析报告，包含趋势判断和买卖信号
        """
        try:
            # 1. 搜索ETF
            etf_list = search_etf_by_name(name)
            if not etf_list or 'error' in etf_list[0]:
                return f"未找到包含'{name}'的ETF"
            
            etf = etf_list[0]
            code = etf['code']
            etf_name = etf['name']
            
            # 2. 获取数据
            daily_df = get_etf_hist_data(code, days=500)
            if daily_df.empty or len(daily_df) < 120:
                return f"数据不足，无法进行波段分析"
            
            weekly_df = resample_to_weekly(daily_df)
            if len(weekly_df) < 60:
                return f"周线数据不足60周，无法进行完整趋势分析"
            
            latest_price = daily_df['close'].iloc[-1]
            
            # 3. 周线趋势分析
            weekly_trend = analyze_weekly_trend(weekly_df)
            
            # 4. 寻找支撑阻力位
            sr_levels = find_support_resistance(weekly_df, 52)
            
            # 5. 日线买入信号分析
            buy_signals = analyze_daily_buy_signals(
                daily_df, 
                weekly_support=sr_levels['nearest_support'],
                weekly_ma20=weekly_trend['ma20']
            )
            
            # 6. 日线卖出信号分析
            sell_signals = analyze_daily_sell_signals(
                daily_df,
                weekly_resistance=sr_levels['nearest_resistance']
            )
            
            # ========== 生成报告 ==========
            output = f"{'='*60}\n"
            output += f"  {etf_name}({code}) 波段交易分析报告\n"
            output += f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            output += f"  当前价格: {latest_price:.4f}\n"
            output += f"{'='*60}\n\n"
            
            # 周线趋势
            output += "【第一步：周线趋势判断】\n"
            output += "-" * 40 + "\n"
            output += f"  趋势类型: {weekly_trend['trend_type']}\n"
            output += f"  趋势强度: {weekly_trend['trend_strength']}分 (-100空头 ~ +100多头)\n"
            output += f"  操作方向: {weekly_trend['operation']}\n\n"
            
            output += "  均线状态:\n"
            output += f"    MA5周:  {weekly_trend['ma5']}\n"
            output += f"    MA20周: {weekly_trend['ma20']} (生命线)\n"
            output += f"    MA60周: {weekly_trend['ma60']} (牛熊线)\n"
            output += f"    价格vs MA20: {weekly_trend['price_vs_ma20']}%\n"
            output += f"    价格vs MA60: {weekly_trend['price_vs_ma60']}%\n"
            output += f"    MA20斜率: {weekly_trend['ma20_slope']}%\n\n"
            
            output += "  MACD状态:\n"
            output += f"    DIF: {weekly_trend['macd_dif']}, DEA: {weekly_trend['macd_dea']}\n"
            output += f"    位置: {weekly_trend['macd_position']}, 状态: {weekly_trend['macd_cross']}\n"
            if weekly_trend['divergence']['type'] != 'none':
                output += f"    ⚠️ {weekly_trend['divergence']['type']}: {weekly_trend['divergence']['description']}\n"
            output += "\n"
            
            output += "  趋势信号:\n"
            for sig in weekly_trend['signals']:
                output += f"    • {sig}\n"
            output += "\n"
            
            # 支撑阻力位
            output += "【关键价位】\n"
            output += "-" * 40 + "\n"
            output += f"  最近支撑位: {sr_levels['nearest_support']:.4f}\n"
            output += f"  最近阻力位: {sr_levels['nearest_resistance']:.4f}\n"
            output += f"  周期最高: {sr_levels['period_high']}\n"
            output += f"  周期最低: {sr_levels['period_low']}\n"
            output += f"  周线MA20: {sr_levels['ma20']}\n"
            output += f"  周线MA60: {sr_levels['ma60']}\n\n"
            
            # 日线买入信号
            output += "【第二步：日线买入信号】\n"
            output += "-" * 40 + "\n"
            output += f"  信号强度: {buy_signals['signal_strength']}分 (70分以上为强信号)\n"
            output += f"  建议: {buy_signals['recommendation']}\n\n"
            
            if buy_signals['signals']:
                output += "  检测到的买入信号:\n"
                for sig in buy_signals['signals']:
                    output += f"    ✓ {sig}\n"
            else:
                output += "  暂无明显买入信号\n"
            output += "\n"
            
            output += "  日线指标:\n"
            ind = buy_signals['indicators']
            output += f"    KDJ: K={ind['kdj_k']}, D={ind['kdj_d']}, J={ind['kdj_j']}\n"
            output += f"    RSI(14): {ind['rsi_14']}\n"
            output += f"    MACD: DIF={ind['macd_dif']}, DEA={ind['macd_dea']}\n"
            output += f"    BOLL%%B: {ind['percent_b']}%\n"
            output += f"    量比: {ind['volume_ratio']}\n\n"
            
            # 日线卖出信号
            output += "【日线卖出/止盈信号】\n"
            output += "-" * 40 + "\n"
            output += f"  信号强度: {sell_signals['signal_strength']}分\n"
            output += f"  建议: {sell_signals['recommendation']}\n\n"
            
            if sell_signals['signals']:
                output += "  检测到的卖出信号:\n"
                for sig in sell_signals['signals']:
                    output += f"    ✗ {sig}\n"
            output += "\n"
            
            # 综合操作建议
            output += "=" * 60 + "\n"
            output += "【综合操作建议】\n"
            output += "=" * 60 + "\n"
            
            trend_strength = weekly_trend['trend_strength']
            buy_strength = buy_signals['signal_strength']
            sell_strength = sell_signals['signal_strength']
            
            if trend_strength >= 60:
                # 强势多头趋势
                if buy_strength >= 70:
                    output += "  ★★★ 强烈买入机会 ★★★\n"
                    output += "  周线强势多头 + 日线强买入信号\n"
                    output += f"  建议: 分批建仓，首次40%仓位\n"
                    output += f"  止损位: {latest_price * 0.96:.4f} (-4%)\n"
                    output += f"  目标位: {sr_levels['nearest_resistance']:.4f}\n"
                elif buy_strength >= 50:
                    output += "  ★★ 较好买入机会 ★★\n"
                    output += "  周线多头趋势 + 日线有买入信号\n"
                    output += f"  建议: 轻仓试探，20%仓位\n"
                    output += f"  止损位: {latest_price * 0.96:.4f} (-4%)\n"
                else:
                    output += "  ★ 趋势向好，等待回调 ★\n"
                    output += "  周线多头趋势，但日线买入信号不足\n"
                    output += f"  建议: 等待价格回调至{weekly_trend['ma20']:.4f}附近\n"
            
            elif trend_strength >= 30:
                # 偏多震荡
                if buy_strength >= 60:
                    output += "  ★ 可考虑轻仓参与 ★\n"
                    output += "  周线偏多震荡 + 日线有买入信号\n"
                    output += f"  建议: 轻仓操作，不超过20%仓位\n"
                    output += f"  止损位: {latest_price * 0.95:.4f} (-5%)\n"
                else:
                    output += "  观望等待\n"
                    output += "  周线方向不够明确，日线信号不足\n"
                    output += "  建议: 等待周线趋势明朗\n"
            
            elif trend_strength <= -60:
                # 强势空头
                output += "  ⚠️ 空仓观望 ⚠️\n"
                output += "  周线空头趋势，不宜做多\n"
                output += "  建议: 耐心等待止跌信号，不要接飞刀\n"
                if weekly_trend['divergence']['type'] == '底背离':
                    output += f"  关注: 周线出现底背离，可能接近底部\n"
            
            elif trend_strength <= -30:
                # 偏空震荡
                output += "  ⚠️ 谨慎观望 ⚠️\n"
                output += "  周线偏空，不建议重仓\n"
                output += "  建议: 减仓或观望，等待趋势反转\n"
            
            else:
                # 震荡整理
                output += "  横盘观望\n"
                output += "  周线方向不明确，震荡整理中\n"
                output += "  建议: 等待突破方向确认\n"
                output += f"  上方阻力: {sr_levels['nearest_resistance']:.4f}\n"
                output += f"  下方支撑: {sr_levels['nearest_support']:.4f}\n"
            
            output += "\n"
            output += "【风险提示】\n"
            output += "  • 任何买入必须设置止损（建议-4%~-5%）\n"
            output += "  • 单只ETF仓位不超过总资金20%-30%\n"
            output += "  • 以上分析仅供参考，不构成投资建议\n"
            
            return output
            
        except Exception as e:
            return f"波段分析失败: {str(e)}"

    @mcp.tool()
    def get_weekly_trend(code: str) -> str:
        """
        获取ETF周线趋势分析
        使用MA20（生命线）和MA60（牛熊分界线）判断中期趋势方向
        
        Args:
            code: ETF代码，如"510300"、"159915"
        
        Returns:
            周线趋势分析报告
        """
        try:
            # 获取ETF名称
            try:
                etf_info = get_cached_etf_spot()
                name_row = etf_info[etf_info['代码'] == code]
                etf_name = name_row['名称'].values[0] if not name_row.empty else code
            except:
                etf_name = code
            
            # 获取数据
            daily_df = get_etf_hist_data(code, days=500)
            if daily_df.empty:
                return f"未能获取ETF {code} 的数据"
            
            weekly_df = resample_to_weekly(daily_df)
            if len(weekly_df) < 60:
                return f"周线数据不足60周，无法进行完整趋势分析"
            
            latest_price = daily_df['close'].iloc[-1]
            
            # 分析趋势
            trend = analyze_weekly_trend(weekly_df)
            
            if 'error' in trend:
                return trend['error']
            
            # 生成报告
            output = f"=== {etf_name}({code}) 周线趋势分析 ===\n\n"
            output += f"当前价格: {latest_price:.4f}\n\n"
            
            output += "【趋势判断】\n"
            output += f"  类型: {trend['trend_type']}\n"
            output += f"  强度: {trend['trend_strength']}分\n"
            output += f"  操作建议: {trend['operation']}\n\n"
            
            output += "【均线系统】\n"
            output += f"  MA5周:  {trend['ma5']}\n"
            output += f"  MA20周: {trend['ma20']} (生命线)\n"
            output += f"  MA60周: {trend['ma60']} (牛熊线)\n"
            output += f"  价格偏离MA20: {trend['price_vs_ma20']}%\n"
            output += f"  价格偏离MA60: {trend['price_vs_ma60']}%\n"
            output += f"  MA20斜率: {trend['ma20_slope']}% (正=向上)\n"
            output += f"  MA60斜率: {trend['ma60_slope']}%\n\n"
            
            output += "【MACD状态】\n"
            output += f"  DIF: {trend['macd_dif']}\n"
            output += f"  DEA: {trend['macd_dea']}\n"
            output += f"  位置: {trend['macd_position']}\n"
            output += f"  状态: {trend['macd_cross']}\n"
            
            if trend['divergence']['type'] != 'none':
                output += f"  ⚠️ 背离: {trend['divergence']['type']} - {trend['divergence']['description']}\n"
            output += "\n"
            
            output += "【趋势信号】\n"
            for sig in trend['signals']:
                output += f"  • {sig}\n"
            
            return output
            
        except Exception as e:
            return f"周线趋势分析失败: {str(e)}"

    @mcp.tool()
    def get_daily_buy_signal(code: str, support_price: float = 0) -> str:
        """
        获取日线买入信号分析
        在周线多头趋势下，寻找日线级别的回调买入机会
        
        检测信号包括：
        - MACD底背离
        - KDJ超卖区金叉
        - RSI超卖
        - 价格回调至关键支撑位
        - 成交量萎缩（卖盘枯竭）
        - 看涨K线形态
        
        Args:
            code: ETF代码
            support_price: 关键支撑位价格（可选）
        
        Returns:
            日线买入信号分析报告
        """
        try:
            # 获取ETF名称
            try:
                etf_info = get_cached_etf_spot()
                name_row = etf_info[etf_info['代码'] == code]
                etf_name = name_row['名称'].values[0] if not name_row.empty else code
            except:
                etf_name = code
            
            # 获取数据
            daily_df = get_etf_hist_data(code, days=120)
            if daily_df.empty or len(daily_df) < 60:
                return f"数据不足，无法分析"
            
            latest_price = daily_df['close'].iloc[-1]
            
            # 分析买入信号
            signals = analyze_daily_buy_signals(
                daily_df,
                weekly_support=support_price if support_price > 0 else None
            )
            
            if 'error' in signals:
                return signals['error']
            
            # 生成报告
            output = f"=== {etf_name}({code}) 日线买入信号 ===\n\n"
            output += f"当前价格: {latest_price:.4f}\n\n"
            
            output += "【信号强度】\n"
            output += f"  综合评分: {signals['signal_strength']}分 (满分100)\n"
            output += f"  建议: {signals['recommendation']}\n\n"
            
            output += "【检测到的买入信号】\n"
            if signals['signals']:
                for sig in signals['signals']:
                    output += f"  ✓ {sig}\n"
            else:
                output += "  暂无明显买入信号\n"
            output += "\n"
            
            output += "【关键指标】\n"
            ind = signals['indicators']
            output += f"  KDJ: K={ind['kdj_k']}, D={ind['kdj_d']}, J={ind['kdj_j']}\n"
            if ind['kdj_j'] < 20:
                output += "       ↳ J值<20，处于超卖区\n"
            output += f"  RSI(14): {ind['rsi_14']}\n"
            if ind['rsi_14'] < 30:
                output += "       ↳ RSI<30，超卖\n"
            output += f"  MACD: DIF={ind['macd_dif']}, DEA={ind['macd_dea']}\n"
            output += f"  BOLL%%B: {ind['percent_b']}%\n"
            if ind['percent_b'] < 20:
                output += "       ↳ 接近布林带下轨\n"
            output += f"  量比: {ind['volume_ratio']}\n"
            if ind['volume_ratio'] < 0.7:
                output += "       ↳ 成交量萎缩，卖盘枯竭\n"
            output += "\n"
            
            output += "【均线参考】\n"
            output += f"  MA5:  {ind['ma5']}\n"
            output += f"  MA10: {ind['ma10']}\n"
            output += f"  MA20: {ind['ma20']}\n"
            output += f"  MA60: {ind['ma60']}\n"
            output += f"  EMA50: {ind['ema50']}\n"
            
            if signals['divergence']['type'] == '底背离':
                output += f"\n⚠️ 重要: 检测到MACD{signals['divergence']['type']}\n"
                output += f"   {signals['divergence']['description']}\n"
            
            return output
            
        except Exception as e:
            return f"买入信号分析失败: {str(e)}"

    @mcp.tool()
    def get_daily_sell_signal(code: str, entry_price: float = 0, resistance_price: float = 0) -> str:
        """
        获取日线卖出/止盈信号分析
        
        检测信号包括：
        - MACD顶背离
        - KDJ超买区死叉
        - RSI超买
        - 价格接近阻力位
        - 跌破短期均线
        
        Args:
            code: ETF代码
            entry_price: 买入价格（用于计算止盈止损，可选）
            resistance_price: 阻力位价格（可选）
        
        Returns:
            日线卖出信号分析报告
        """
        try:
            # 获取ETF名称
            try:
                etf_info = get_cached_etf_spot()
                name_row = etf_info[etf_info['代码'] == code]
                etf_name = name_row['名称'].values[0] if not name_row.empty else code
            except:
                etf_name = code
            
            # 获取数据
            daily_df = get_etf_hist_data(code, days=120)
            if daily_df.empty or len(daily_df) < 30:
                return f"数据不足，无法分析"
            
            latest_price = daily_df['close'].iloc[-1]
            
            # 分析卖出信号
            signals = analyze_daily_sell_signals(
                daily_df,
                entry_price=entry_price if entry_price > 0 else None,
                weekly_resistance=resistance_price if resistance_price > 0 else None
            )
            
            if 'error' in signals:
                return signals['error']
            
            # 生成报告
            output = f"=== {etf_name}({code}) 日线卖出信号 ===\n\n"
            output += f"当前价格: {latest_price:.4f}\n"
            
            if entry_price > 0:
                profit_pct = (latest_price - entry_price) / entry_price * 100
                output += f"买入价格: {entry_price:.4f}\n"
                output += f"当前盈亏: {profit_pct:.2f}%\n"
            output += "\n"
            
            output += "【信号强度】\n"
            output += f"  综合评分: {signals['signal_strength']}分\n"
            output += f"  建议: {signals['recommendation']}\n\n"
            
            output += "【检测到的卖出信号】\n"
            if signals['signals']:
                for sig in signals['signals']:
                    output += f"  ✗ {sig}\n"
            else:
                output += "  暂无明显卖出信号\n"
            output += "\n"
            
            if signals['stop_loss'] or signals['take_profit']:
                output += "【止盈止损建议】\n"
                if signals['stop_loss']:
                    output += f"  止损位: {signals['stop_loss']:.4f}\n"
                if signals['take_profit']:
                    output += f"  移动止盈位: {signals['take_profit']:.4f}\n"
                output += "\n"
            
            output += "【关键指标】\n"
            ind = signals['indicators']
            output += f"  KDJ: K={ind['kdj_k']}, D={ind['kdj_d']}, J={ind['kdj_j']}\n"
            if ind['kdj_j'] > 80:
                output += "       ↳ J值>80，处于超买区\n"
            output += f"  RSI(14): {ind['rsi_14']}\n"
            if ind['rsi_14'] > 70:
                output += "       ↳ RSI>70，超买\n"
            output += f"  MACD: DIF={ind['macd_dif']}, DEA={ind['macd_dea']}\n"
            output += f"  BOLL%%B: {ind['percent_b']}%\n"
            if ind['percent_b'] > 80:
                output += "       ↳ 接近布林带上轨\n"
            
            if signals['divergence']['type'] == '顶背离':
                output += f"\n⚠️ 重要: 检测到MACD{signals['divergence']['type']}\n"
                output += f"   {signals['divergence']['description']}\n"
            
            return output
            
        except Exception as e:
            return f"卖出信号分析失败: {str(e)}"

    @mcp.tool()
    def get_support_resistance(code: str, weeks: int = 52) -> str:
        """
        获取ETF的关键支撑阻力位
        
        Args:
            code: ETF代码
            weeks: 分析周期（周数），默认52周
        
        Returns:
            支撑阻力位分析报告
        """
        try:
            # 获取ETF名称
            try:
                etf_info = get_cached_etf_spot()
                name_row = etf_info[etf_info['代码'] == code]
                etf_name = name_row['名称'].values[0] if not name_row.empty else code
            except:
                etf_name = code
            
            # 获取数据
            daily_df = get_etf_hist_data(code, days=weeks * 7)
            if daily_df.empty:
                return f"未能获取ETF {code} 的数据"
            
            weekly_df = resample_to_weekly(daily_df)
            latest_price = daily_df['close'].iloc[-1]
            
            # 分析支撑阻力
            sr = find_support_resistance(weekly_df, weeks)
            
            # 生成报告
            output = f"=== {etf_name}({code}) 支撑阻力位分析 ===\n\n"
            output += f"当前价格: {latest_price:.4f}\n"
            output += f"分析周期: 近{weeks}周\n\n"
            
            output += "【关键价位】\n"
            output += f"  周期最高: {sr['period_high']}\n"
            output += f"  周期最低: {sr['period_low']}\n\n"
            
            output += "【阻力位】(由近到远)\n"
            for i, r in enumerate(sr['resistances'][:3], 1):
                distance = (r - latest_price) / latest_price * 100
                output += f"  {i}. {r:.4f} (距当前+{distance:.1f}%)\n"
            output += "\n"
            
            output += "【支撑位】(由近到远)\n"
            for i, s in enumerate(sr['supports'][:3], 1):
                distance = (latest_price - s) / latest_price * 100
                output += f"  {i}. {s:.4f} (距当前-{distance:.1f}%)\n"
            output += "\n"
            
            output += "【均线支撑阻力】\n"
            ma20_pos = "支撑" if latest_price > sr['ma20'] else "阻力"
            ma60_pos = "支撑" if latest_price > sr['ma60'] else "阻力"
            output += f"  周线MA20: {sr['ma20']} ({ma20_pos})\n"
            output += f"  周线MA60: {sr['ma60']} ({ma60_pos})\n"
            
            return output
            
        except Exception as e:
            return f"支撑阻力分析失败: {str(e)}"

    @mcp.tool()
    def screen_swing_opportunities(category: str = "index") -> str:
        """
        筛选波段交易机会
        扫描符合"周线多头+日线回调"条件的ETF
        
        Args:
            category: ETF类别
                - "index": 宽基指数ETF（沪深300、创业板等）
                - "industry": 行业ETF
                - "cross_border": 跨境ETF
        
        Returns:
            符合条件的ETF列表及其信号强度
        """
        try:
            # 定义筛选标的
            targets = {
                "index": ["沪深300", "创业板", "科创50", "中证500", "上证50"],
                "industry": ["医药", "消费", "新能源", "半导体", "军工", "银行"],
                "cross_border": ["纳斯达克", "标普500", "恒生科技", "恒生"]
            }
            
            etf_names = targets.get(category, targets["index"])
            
            output = f"=== 波段交易机会筛选 ({category}) ===\n"
            output += f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            output += f"筛选条件: 周线多头趋势 + 日线有买入信号\n\n"
            
            opportunities = []
            
            for name in etf_names:
                try:
                    etf_list = search_etf_by_name(name)
                    if not etf_list or 'error' in etf_list[0]:
                        continue
                    
                    etf = etf_list[0]
                    code = etf['code']
                    
                    # 获取数据
                    daily_df = get_etf_hist_data(code, days=400)
                    if len(daily_df) < 120:
                        continue
                    
                    weekly_df = resample_to_weekly(daily_df)
                    if len(weekly_df) < 60:
                        continue
                    
                    # 分析周线趋势
                    trend = analyze_weekly_trend(weekly_df)
                    if 'error' in trend:
                        continue
                    
                    # 只关注多头或偏多趋势
                    if trend['trend_strength'] < 30:
                        continue
                    
                    # 分析日线买入信号
                    buy_signals = analyze_daily_buy_signals(daily_df, weekly_ma20=trend['ma20'])
                    
                    # 记录机会
                    opportunities.append({
                        'name': etf['name'],
                        'code': code,
                        'price': daily_df['close'].iloc[-1],
                        'trend_type': trend['trend_type'],
                        'trend_strength': trend['trend_strength'],
                        'buy_strength': buy_signals['signal_strength'],
                        'signals': buy_signals['signals'][:3]  # 只取前3个信号
                    })
                    
                except Exception:
                    continue
            
            # 按综合评分排序
            opportunities.sort(key=lambda x: x['trend_strength'] + x['buy_strength'], reverse=True)
            
            if not opportunities:
                output += "暂无符合条件的ETF\n"
                output += "（需要周线趋势强度>=30分）\n"
                return output
            
            output += f"找到 {len(opportunities)} 个机会:\n\n"
            
            for i, opp in enumerate(opportunities, 1):
                total_score = opp['trend_strength'] + opp['buy_strength']
                output += f"{i}. {opp['name']}({opp['code']})\n"
                output += f"   价格: {opp['price']:.4f}\n"
                output += f"   周线趋势: {opp['trend_type']} ({opp['trend_strength']}分)\n"
                output += f"   日线买入信号: {opp['buy_strength']}分\n"
                output += f"   综合评分: {total_score}分\n"
                
                if opp['signals']:
                    output += "   信号:\n"
                    for sig in opp['signals']:
                        output += f"     • {sig}\n"
                output += "\n"
            
            output += "【说明】\n"
            output += "  • 综合评分 = 周线趋势分 + 日线买入信号分\n"
            output += "  • 评分越高，机会越好\n"
            output += "  • 建议优先关注综合评分>100的标的\n"
            
            return output
            
        except Exception as e:
            return f"筛选失败: {str(e)}"
