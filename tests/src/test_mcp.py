from __future__ import annotations

import asyncio
import builtins
import os
import sys
import types
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

    def test_load_mcp_server_catalog_reads_local_prefixed_descriptors_without_parsing_agent_yaml(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "agent.yaml").write_text(
                "\n".join(
                    [
                        'name: "Workspace Agent"',
                        "model:",
                        '  name: "gpt-4o"',
                        '  provider: "openai"',
                        '  api-key: "secret"',
                        "prompt: |",
                        "  Inspect files.",
                    ]
                ),
                encoding="utf-8",
            )
            (temp_path / "mcp-file-system.yaml").write_text(
                "\n".join(
                    [
                        'name: "file-system"',
                        'transport: "stdio"',
                        'command: "uvx"',
                    ]
                ),
                encoding="utf-8",
            )

            with patch("ftry.Mcp.Path.cwd", return_value=temp_path):
                catalog = mcp_module.Mcp.load_catalog()

        self.assertEqual([config.name for config in catalog], ["file-system"])

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

    def test_open_connections_discovers_runtime_capabilities(self) -> None:
        reset_fakes()
        FakeMCPStdioTool.discovery_tools = [
            types.SimpleNamespace(
                name="write_file",
                title="Write file",
                description="Write a file to disk.",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        FakeMCPStdioTool.discovery_prompts = [
            types.SimpleNamespace(
                name="summarize_path",
                description="Summarize a file.",
                arguments=[types.SimpleNamespace(name="path", description="Target path", required=True)],
            )
        ]
        FakeMCPStdioTool.discovery_resources = [
            types.SimpleNamespace(name="workspace-root", uri="file:///workspace", description="Workspace root")
        ]
        FakeMCPStdioTool.discovery_resource_templates = [
            types.SimpleNamespace(name="workspace-file", uriTemplate="file:///workspace/{path}", description="Workspace file")
        ]

        async def open_connections() -> tuple[object, ...]:
            async with AsyncExitStack() as exit_stack:
                return await mcp_module.Mcp.open_connections(
                    (mcp_module.Mcp(mcp_module.McpConfig(name="file-system", transport="stdio", command="uvx")),),
                    exit_stack=exit_stack,
                )

        with patch.dict(sys.modules, make_fake_agent_framework_modules(), clear=False):
            connections = asyncio.run(open_connections())

        self.assertEqual(len(connections), 1)
        capabilities = connections[0].capabilities
        self.assertEqual(capabilities.server_name, "file-system")
        self.assertEqual(capabilities.tools[0].name, "write_file")
        self.assertIn('"path"', capabilities.tools[0].input_schema_summary or "")
        self.assertEqual(capabilities.prompts[0].name, "summarize_path")
        self.assertIn("path: Target path (required)", capabilities.prompts[0].arguments_summary or "")
        self.assertEqual(capabilities.resources[0].uri, "file:///workspace")
        self.assertEqual(capabilities.resource_templates[0].uri_template, "file:///workspace/{path}")

    def test_render_runtime_context_reports_partial_discovery_warnings(self) -> None:
        connection = mcp_module.McpRuntimeConnection(
            mcp=mcp_module.Mcp(mcp_module.McpConfig(name="demo", transport="stdio", command="uvx")),
            tool=object(),
            capabilities=mcp_module.McpServerCapabilities(
                server_name="demo",
                transport="stdio",
                warnings=("Runtime MCP session does not expose resources.",),
            ),
        )

        rendered = mcp_module.Mcp.render_runtime_context((connection,))

        self.assertIsNotNone(rendered)
        assert rendered is not None
        self.assertIn("<McpRuntimeCapabilities>", rendered)
        self.assertIn("Runtime MCP session does not expose resources.", rendered)

    def test_discover_capabilities_reports_missing_live_session(self) -> None:
        capabilities = asyncio.run(
            mcp_module.Mcp(
                mcp_module.McpConfig(name="demo", transport="stdio", command="uvx")
            ).discover_capabilities(object())
        )

        self.assertEqual(capabilities.server_name, "demo")
        self.assertIn("Live MCP session metadata is not exposed by the runtime tool.", capabilities.warnings)

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
