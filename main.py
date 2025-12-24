"""
指数基金ETF辅助MCP服务
基于akshare和aktools实现，提供ETF相关技术指标和数据分析工具

模块结构:
- cache.py: 缓存管理
- indicators.py: 技术指标计算
- data.py: 数据获取
- tools.py: MCP工具定义
- main.py: 入口文件

使用方式:
- stdio模式: python main.py
- HTTP模式: python main.py --http [--host HOST] [--port PORT]
"""

import argparse
from mcp.server.fastmcp import FastMCP
from tools import register_tools

# 创建MCP服务实例
mcp = FastMCP("ETF辅助服务")

# 注册所有工具
register_tools(mcp)


# ==================== 启动服务 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF MCP 服务")
    parser.add_argument("--http", action="store_true", help="使用 Streamable HTTP 模式")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP 服务监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP 服务端口 (默认: 8000)")
    
    args = parser.parse_args()
    
    if args.http:
        # Streamable HTTP 模式 - 使用 uvicorn 自定义 host/port
        import uvicorn
        print(f"启动 Streamable HTTP 服务: http://{args.host}:{args.port}/mcp")
        app = mcp.streamable_http_app()
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # 默认 stdio 模式
        mcp.run()
