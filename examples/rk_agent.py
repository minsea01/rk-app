#!/usr/bin/env python3
"""
RK3588 开发助手 - 使用配置文件
完全独立于 Claude Code CLI，使用自定义 API endpoint
"""
import asyncio
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import os
from pathlib import Path

class RKAgent:
    def __init__(self, config_path: str = None):
        """
        初始化 RK3588 Agent

        Args:
            config_path: 配置文件路径，默认为 ../.agent_config.json
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / ".agent_config.json"

        # 读取配置
        with open(config_path) as f:
            self.config = json.load(f)

        # 初始化 Anthropic 客户端（使用自定义 API）
        self.client = Anthropic(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )

        self.mcp_sessions = {}
        self.tools = []

        print(f"✓ 使用配置: {config_path}")
        print(f"  API Endpoint: {self.config['base_url']}")

    async def initialize_mcp(self):
        """初始化所有 MCP 服务器"""
        for server_name, server_config in self.config["mcp_servers"].items():
            try:
                print(f"正在连接 MCP 服务器: {server_name}...", end=" ")

                server_params = StdioServerParameters(
                    command=server_config["command"],
                    args=server_config["args"],
                    env=server_config.get("env", {})
                )

                transport = stdio_client(server_params)
                read_stream, write_stream = await transport.__aenter__()

                session_context = ClientSession(read_stream, write_stream)
                session = await session_context.__aenter__()
                await session.initialize()

                # 获取工具列表
                tools_result = await session.list_tools()
                server_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    }
                    for tool in tools_result.tools
                ]

                self.mcp_sessions[server_name] = session
                self.tools.extend(server_tools)

                print(f"✓ ({len(server_tools)} 个工具)")

            except Exception as e:
                print(f"✗ 失败: {e}")

        print(f"\n总计加载 {len(self.tools)} 个工具")

    async def chat(self, user_message: str, max_iterations: int = 10, verbose: bool = True):
        """
        与 Agent 对话

        Args:
            user_message: 用户消息
            max_iterations: 最大工具调用迭代次数
            verbose: 是否显示详细信息
        """
        messages = [{"role": "user", "content": user_message}]

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n[迭代 {iteration + 1}/{max_iterations}]")

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                tools=self.tools,
                messages=messages
            )

            # 输出助手回复
            assistant_text = []
            for content in response.content:
                if content.type == "text":
                    assistant_text.append(content.text)
                    if verbose:
                        print(f"\n🤖 Assistant: {content.text}")

            # 检查是否需要调用工具
            if response.stop_reason == "tool_use":
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        if verbose:
                            print(f"\n🔧 调用工具: {content_block.name}")
                            print(f"   参数: {json.dumps(content_block.input, ensure_ascii=False)}")

                        try:
                            # 找到对应的 MCP 服务器并调用工具
                            result = None
                            for session in self.mcp_sessions.values():
                                try:
                                    result = await session.call_tool(
                                        content_block.name,
                                        arguments=content_block.input
                                    )
                                    break
                                except:
                                    continue

                            if result:
                                result_text = str(result.content[0].text) if result.content else "OK"
                                if verbose:
                                    preview = result_text[:200] + "..." if len(result_text) > 200 else result_text
                                    print(f"   结果: {preview}")

                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": content_block.id,
                                    "content": result_text
                                })
                            else:
                                raise Exception("工具未找到")

                        except Exception as e:
                            if verbose:
                                print(f"   ✗ 错误: {e}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": f"Error: {str(e)}",
                                "is_error": True
                            })

                messages.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                if verbose:
                    print(f"\n✓ 对话结束 (原因: {response.stop_reason})")
                break

        return assistant_text

    async def interactive(self):
        """交互式对话模式"""
        print("\n" + "="*60)
        print("🤖 RK3588 开发助手 - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n再见！")
                    break

                await self.chat(user_input)

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n✗ 错误: {e}")

    async def close(self):
        """关闭所有连接"""
        for session in self.mcp_sessions.values():
            try:
                await session.__aexit__(None, None, None)
            except:
                pass

async def main():
    """主函数"""
    import sys

    # 创建 Agent
    agent = RKAgent()

    try:
        # 初始化 MCP
        await agent.initialize_mcp()

        # 根据命令行参数决定模式
        if len(sys.argv) > 1:
            # 命令行模式
            user_query = " ".join(sys.argv[1:])
            print(f"\n查询: {user_query}")
            await agent.chat(user_query)
        else:
            # 交互模式
            await agent.interactive()

    except FileNotFoundError:
        print("\n❌ 错误: 配置文件不存在")
        print("   请创建 .agent_config.json 文件")
        print("   示例: cp .agent_config.json.example .agent_config.json")
    except KeyError as e:
        print(f"\n❌ 配置文件格式错误: 缺少字段 {e}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
