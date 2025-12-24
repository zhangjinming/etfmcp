"""
指数基金ETF辅助MCP服务
基于akshare和aktools实现，提供ETF相关技术指标和数据分析工具

模块结构:
- cache.py: 缓存管理
- indicators.py: 技术指标计算
- data.py: 数据获取
- tools.py: MCP工具定义
- main.py: 入口文件
"""

from mcp.server.fastmcp import FastMCP
from tools import register_tools

# 创建MCP服务实例
mcp = FastMCP("ETF辅助服务")

# 注册所有工具
register_tools(mcp)


# ==================== 启动服务 ====================

if __name__ == "__main__":
    mcp.run()
