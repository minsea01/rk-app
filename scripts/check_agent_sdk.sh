#!/bin/bash
echo "=== Anthropic Agent SDK 配置检查 ==="

# 1. Python 版本
echo -n "1. Python 版本: "
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [ "$(echo "$python_version >= 3.10" | bc -l 2>/dev/null || python3 -c "print(float('$python_version') >= 3.10)")" = "True" ] || [ "$(echo "$python_version" | awk '{if ($1 >= 3.10) print 1; else print 0}')" = "1" ]; then
    echo "✓ $python_version"
else
    echo "✗ $python_version (需要 3.10+)"
fi

# 2. Claude Code CLI
echo -n "2. Claude Code CLI: "
if command -v claude &> /dev/null; then
    echo "✓ $(claude --version 2>&1 | head -1)"
else
    echo "✗ 未安装"
fi

# 3. API Key
echo -n "3. ANTHROPIC_API_KEY: "
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✓ 已设置 (${ANTHROPIC_API_KEY:0:10}...)"
else
    echo "✗ 未设置"
fi

# 4. Agent SDK
echo -n "4. Claude Agent SDK: "
if python3 -c "import claude_agent_sdk" 2>/dev/null; then
    sdk_version=$(python3 -c "import claude_agent_sdk; print(getattr(claude_agent_sdk, '__version__', 'unknown'))" 2>/dev/null)
    echo "✓ 已安装 ($sdk_version)"
else
    echo "✗ 未安装"
fi

# 5. MCP SDK
echo -n "5. MCP Python SDK: "
if python3 -c "import mcp" 2>/dev/null; then
    echo "✓ 已安装"
else
    echo "✗ 未安装"
fi

# 6. MCP 服务器
echo -n "6. MCP 服务器配置: "
mcp_count=$(python3 -c "import json; print(len(json.load(open('$HOME/.claude.json')).get('mcpServers', {})))" 2>/dev/null || echo "0")
if [ "$mcp_count" -gt 0 ]; then
    echo "✓ $mcp_count 个已配置"
else
    echo "✗ 未配置"
fi

echo ""
echo "=== 后续步骤 ==="
need_steps=false

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❗ 1. 获取 API 密钥:"
    echo "      访问: https://console.anthropic.com/settings/keys"
    echo "      设置: export ANTHROPIC_API_KEY='sk-ant-api03-...'"
    echo "      永久: echo 'export ANTHROPIC_API_KEY=\"your-key\"' >> ~/.bashrc"
    need_steps=true
fi

if ! python3 -c "import claude_agent_sdk" 2>/dev/null; then
    echo "❗ 2. 安装 Agent SDK:"
    echo "      pip install claude-agent-sdk"
    need_steps=true
fi

if [ "$need_steps" = false ]; then
    echo "✅ 所有依赖已就绪！"
    echo ""
    echo "📝 可以运行示例代码："
    echo "   python3 test_agent_sdk.py"
fi
