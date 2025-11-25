#!/usr/bin/env python3
"""
RK3588 简单 Agent - 仅使用 Anthropic API（不依赖 MCP）
测试自定义 API endpoint 是否正常工作
"""
import asyncio
from anthropic import Anthropic
import json
from pathlib import Path

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / ".agent_config.json"
    with open(config_path) as f:
        return json.load(f)

def main():
    """主函数"""
    import sys

    # 加载配置
    config = load_config()

    print(f"✓ 配置加载成功")
    print(f"  API Endpoint: {config['base_url']}")
    print(f"  API Key: {config['api_key'][:20]}...\n")

    # 初始化 Anthropic 客户端
    client = Anthropic(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )

    # 获取用户查询
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "你好！请简单介绍一下你自己。"

    print(f"查询: {user_query}\n")
    print("="*60)

    try:
        # 调用 API
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_query}
            ]
        )

        # 输出结果
        print(f"\n🤖 Assistant:\n")
        for content in message.content:
            if content.type == "text":
                print(content.text)

        print("\n" + "="*60)
        print(f"✓ API 调用成功")
        print(f"  模型: {message.model}")
        print(f"  使用 tokens: {message.usage.input_tokens} 输入 + {message.usage.output_tokens} 输出")

    except (ConnectionError, ValueError, RuntimeError) as e:
        print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    main()
