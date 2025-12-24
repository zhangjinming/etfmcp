# ETF MCP 服务

基于 [akshare](https://github.com/akfamily/akshare) 和 [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) 实现的指数基金 ETF 辅助服务，为 AI Agent 提供 ETF 相关技术指标和数据分析工具。

## 功能特性

- **ETF 搜索与查询**：按名称搜索 ETF，获取实时行情
- **技术指标分析**：BOLL、RSI、MACD、KDJ、MA、EMA、ATR、OBV 等
- **趋势分析**：基于周线数据的多周期趋势评分系统
- **市场概览**：中国/全球主要指数实时行情
- **宏观数据**：M2、出口、外汇储备等经济指标
- **数据缓存**：内置智能缓存，减少重复请求

## 安装

### 依赖安装

```bash
pip install akshare fastmcp pandas numpy
```

### 克隆项目

```bash
git clone https://github.com/yourusername/etfmcp.git
cd etfmcp
```

## 使用方法

### stdio 模式（默认）

```bash
python main.py
```

### Streamable HTTP 模式

```bash
# 默认配置 (0.0.0.0:8000/mcp)
python main.py --http

# 自定义配置
python main.py --http --host 127.0.0.1 --port 9000
```

### 在 Claude Desktop 中配置

**stdio 模式：**

```json
{
  "mcpServers": {
    "etf": {
      "command": "python",
      "args": ["/path/to/etfmcp/main.py"]
    }
  }
}
```

**Streamable HTTP 模式：**

```json
{
  "mcpServers": {
    "etf": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## MCP 工具列表

| 工具名称 | 功能描述 |
|---------|---------|
| `search_etf` | 按名称搜索 ETF |
| `get_etf_realtime_info` | 获取 ETF 实时行情 |
| `get_etf_technical_indicators` | 获取技术指标（日线/周线） |
| `analyze_etf_trend` | 多周期趋势分析（含评分） |
| `get_etf_comprehensive_analysis` | ETF 综合分析报告 |
| `get_index_realtime` | 指数实时行情（中国/全球） |
| `get_index_history` | 指数历史数据 |
| `get_market_overview` | 市场概览 |
| `get_etf_list_by_category` | 按类别获取 ETF 列表 |
| `compare_etfs` | 多 ETF 对比 |
| `get_etf_performance_ranking` | ETF 涨跌幅排行 |
| `get_multi_etf_indicators` | 批量 ETF 指标查询 |
| `get_macro_economic_data` | 宏观经济数据 |
| `get_economic_calendar` | 经济事件日历 |
| `clear_data_cache` | 清除数据缓存 |
| `get_data_cache_stats` | 获取缓存状态 |

## 项目结构

```
etfmcp/
├── main.py          # 入口文件，创建 MCP 服务实例
├── cache.py         # 缓存管理模块
├── indicators.py    # 技术指标计算模块
├── data.py          # 数据获取模块
├── tools.py         # MCP 工具定义模块
└── test_mcp_tools.py # 测试文件
```

## 技术指标评分体系

`analyze_etf_trend` 工具基于周线数据分析，适合中长期趋势判断：

| 指标 | 分值范围 | 说明 |
|-----|---------|------|
| BOLL 位置 | ±35分 | 主要指标，超卖/超买判断 |
| 成交量 | ±20分 | 量价配合分析 |
| RSI | ±15分 | 超买超卖判断 |
| MACD | ±15分 | 金叉/死叉信号 |
| 均线排列 | ±15分 | 多头/空头排列 |

**评分解读**：
- `> 30`：强势上涨趋势
- `10 ~ 30`：温和上涨
- `-10 ~ 10`：横盘整理
- `-30 ~ -10`：温和下跌
- `< -30`：强势下跌趋势

## 缓存配置

| 数据类型 | 缓存时间 |
|---------|---------|
| ETF 实时行情 | 60秒 |
| ETF 历史数据 | 5分钟 |
| 指数实时行情 | 60秒 |
| 指数历史数据 | 5分钟 |
| 宏观数据 | 1小时 |
| 经济日历 | 1小时 |

## 测试

```bash
# 运行所有测试
python test_mcp_tools.py

# 只运行单元测试（无需网络）
python test_mcp_tools.py --unit

# 只运行集成测试（需要网络）
python test_mcp_tools.py --integration
```

## License

MIT License
