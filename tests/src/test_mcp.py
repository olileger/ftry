from __future__ import annotations

import asyncio
import builtins
import os
import sys
import unittest
from contextlib import AsyncExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ftry.Mcp as mcp_module
from tests.src.testsupport import (
    FakeMCPStdioTool,
    FakeMCPStreamableHTTPTool,
    FakeMCPWebsocketTool,
    make_fake_agent_framework_modules,
    reset_fakes,
)


class McpTests(unittest.TestCase):
    def test_load_mcp_server_catalog_parses_supported_descriptor_shapes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_dir = temp_path / "mcp"
            registry_dir.mkdir()
            (registry_dir / "file-system.yaml").write_text(
                "\n".join(
                    [
                        'name: "file-system"',
                        'transport: "stdio"',
                        "description: |",
                        "  Local filesystem access.",
                        'command: "uvx"',
                        "args:",
                        '  - "mcp-server-filesystem"',
                        '  - "C:\\\\work"',
                        "env:",
                        '  ROOT: "env:MCP_ROOT"',
                    ]
                ),
                encoding="utf-8",
            )
            (registry_dir / "github.yaml").write_text(
                "\n".join(
                    [
                        'name: "github"',
                        'transport: "http"',
                        'url: "https://example.com/mcp"',
                        "headers:",
                        '  Authorization: "env:GITHUB_TOKEN"',
                        "allowed-tools:",
                        '  - "search_repositories"',
                    ]
                ),
                encoding="utf-8",
            )

            with patch("ftry.Mcp.Path.cwd", return_value=temp_path):
                catalog = mcp_module.Mcp.load_catalog()

        self.assertEqual([config.name for config in catalog], ["file-system", "github"])
        self.assertEqual(catalog[0].transport, "stdio")
        self.assertEqual(catalog[0].args, ("mcp-server-filesystem", r"C:\work"))
        self.assertEqual(catalog[1].allowed_tools, ("search_repositories",))

    def test_resolve_mcp_server_configs_rejects_missing_descriptors(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with patch("ftry.Mcp.Path.cwd", return_value=temp_path):
                with self.assertRaisesRegex(mcp_module.FtryCliError, "not found"):
                    mcp_module.Mcp.resolve_configs(["missing-server"])

    def test_create_mcp_tool_builds_framework_specific_tools(self) -> None:
        reset_fakes()
        with (
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
            patch.dict(os.environ, {"MCP_ROOT": r"C:\repo", "GITHUB_TOKEN": "bearer secret"}, clear=False),
        ):
            stdio_tool = mcp_module.Mcp(
                mcp_module.McpConfig(
                    name="file-system",
                    transport="stdio",
                    command="uvx",
                    args=("mcp-server-filesystem", r"C:\repo"),
                    env={"ROOT": "env:MCP_ROOT"},
                )
            ).create_tool()
            http_tool = mcp_module.Mcp(
                mcp_module.McpConfig(
                    name="github",
                    transport="http",
                    url="https://example.com/mcp",
                    headers={"Authorization": "env:GITHUB_TOKEN"},
                    allowed_tools=("search_repositories",),
                )
            ).create_tool()
            websocket_tool = mcp_module.Mcp(
                mcp_module.McpConfig(
                    name="realtime",
                    transport="websocket",
                    url="wss://example.com/mcp",
                    headers={"Authorization": "env:GITHUB_TOKEN"},
                )
            ).create_tool()
            self.assertIsInstance(stdio_tool, FakeMCPStdioTool)
            self.assertEqual(stdio_tool.kwargs["env"], {"ROOT": r"C:\repo"})
            self.assertIsInstance(http_tool, FakeMCPStreamableHTTPTool)
            self.assertEqual(http_tool.kwargs["allowed_tools"], ["search_repositories"])
            self.assertEqual(http_tool.kwargs["header_provider"]({}), {"Authorization": "bearer secret"})
            self.assertIsInstance(websocket_tool, FakeMCPWebsocketTool)
            self.assertEqual(websocket_tool.kwargs["headers"], {"Authorization": "bearer secret"})

    def test_open_mcp_tools_enters_and_closes_tool_contexts(self) -> None:
        reset_fakes()
        mcps = (
            mcp_module.Mcp(mcp_module.McpConfig(name="file-system", transport="stdio", command="uvx")),
            mcp_module.Mcp(mcp_module.McpConfig(name="github", transport="http", url="https://example.com/mcp")),
        )

        async def open_tools() -> tuple[object, ...]:
            async with AsyncExitStack() as exit_stack:
                return await mcp_module.Mcp.open_tools(mcps, exit_stack=exit_stack)

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            opened_tools = asyncio.run(open_tools())

        self.assertEqual(len(opened_tools), 2)
        self.assertEqual(len(FakeMCPStdioTool.entered_tools), 1)
        self.assertEqual(len(FakeMCPStdioTool.closed_tools), 1)
        self.assertEqual(len(FakeMCPStreamableHTTPTool.entered_tools), 1)
        self.assertEqual(len(FakeMCPStreamableHTTPTool.closed_tools), 1)

    def test_create_mcp_tool_requires_python_mcp_package(self) -> None:
        reset_fakes()
        original_import = builtins.__import__

        def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mcp":
                raise ImportError("No module named 'mcp'")
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False),
            patch("builtins.__import__", side_effect=failing_import),
        ):
            with self.assertRaisesRegex(mcp_module.FtryCliError, "The Python package `mcp` is required"):
                mcp_module.Mcp(
                    mcp_module.McpConfig(name="file-system", transport="stdio", command="uvx")
                ).create_tool()


if __name__ == "__main__":
    unittest.main()
