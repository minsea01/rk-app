#!/usr/bin/env python3
"""
测试 API endpoint 和可用模型
"""
from anthropic import Anthropic
import json
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent / ".agent_config.json"
    with open(config_path) as f:
        return json.load(f)

def test_models():
    config = load_config()

    print(f"测试 API: {config['base_url']}")
    print(f"API Key: {config['api_key'][:20]}...\n")

    client = Anthropic(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )

    # 常见的 Claude 模型列表
    models_to_test = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    print("测试可用模型:\n")
    print("="*60)

    successful_models = []

    for model in models_to_test:
        print(f"\n测试: {model}...", end=" ")
        try:
            message = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}]
            )
            print("✓ 成功")
            successful_models.append(model)

        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "无权访问" in error_msg:
                print("✗ 无权限")
            elif "404" in error_msg or "not found" in error_msg.lower():
                print("✗ 模型不存在")
            else:
                print(f"✗ 错误: {error_msg[:50]}")

    print("\n" + "="*60)
    if successful_models:
        print(f"\n✓ 可用模型 ({len(successful_models)} 个):")
        for model in successful_models:
            print(f"  - {model}")
    else:
        print("\n❌ 未找到可用模型")
        print("\n建议:")
        print("1. 检查 API key 是否正确")
        print("2. 确认 API endpoint 是否支持 Claude 模型")
        print("3. 联系 API 提供商确认可用模型列表")

if __name__ == "__main__":
    test_models()
