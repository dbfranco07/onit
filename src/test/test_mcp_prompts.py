"""Tests for src/mcp/prompts/prompts.py — assistant_instruction."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import the module and get the assistant_instruction function.
# Depending on the fastmcp version, the @prompt() decorator may return
# the original function directly or wrap it in a FunctionPrompt with .fn.
import src.mcp.prompts.prompts as prompts_mod

_decorated = prompts_mod.assistant_instruction
_assistant_fn = getattr(_decorated, "fn", _decorated)


class TestAssistantInstruction:
    @pytest.mark.asyncio
    async def test_basic_instruction(self):
        result = await _assistant_fn(task="What is 2+2?", session_id="test-session")
        assert "What is 2+2?" in result
        assert "test-session" in result

    @pytest.mark.asyncio
    async def test_generates_session_id_if_none(self):
        result = await _assistant_fn(task="test task")
        assert "test task" in result
        assert "onit" in result

    @pytest.mark.asyncio
    async def test_includes_data_path(self):
        result = await _assistant_fn(task="test", session_id="sid")
        assert "data" in result

    @pytest.mark.asyncio
    async def test_custom_template(self, tmp_path):
        template_content = {
            "instruction_template": "Custom: {task} in {data_path} for {session_id}"
        }
        template_file = tmp_path / "custom.yaml"
        template_file.write_text(yaml.dump(template_content))

        result = await _assistant_fn(
            task="my task",
            session_id="s1",
            template_path=str(template_file),
        )
        assert "Custom: my task" in result

    @pytest.mark.asyncio
    async def test_invalid_template_uses_default(self, tmp_path):
        template_file = tmp_path / "empty.yaml"
        template_file.write_text(yaml.dump({"other_key": "value"}))

        result = await _assistant_fn(
            task="fallback test",
            session_id="s2",
            template_path=str(template_file),
        )
        assert "fallback test" in result
        assert "step by step" in result

    @pytest.mark.asyncio
    async def test_nonexistent_template_uses_default(self):
        result = await _assistant_fn(
            task="no template",
            session_id="s3",
            template_path="/nonexistent/template.yaml",
        )
        assert "no template" in result
        assert "step by step" in result

    @pytest.mark.asyncio
    async def test_file_server_url_appended(self):
        result = await _assistant_fn(
            task="create report",
            session_id="s4",
            file_server_url="http://192.168.1.100:9000",
        )
        assert "http://192.168.1.100:9000" in result
        assert "uploads" in result
        assert "callback_url" in result

    @pytest.mark.asyncio
    async def test_no_file_server_url(self):
        result = await _assistant_fn(
            task="simple task",
            session_id="s5",
            file_server_url=None,
        )
        assert "uploads" not in result


class TestRoboticsPromptTemplate:
    def test_clear_path_between_rule_is_precise_and_one_pass(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / "mcp"
            / "prompts"
            / "prompt_templates"
            / "assistant_robotics.yaml"
        )
        config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        template = config["instruction_template"]

        assert "delta = ((H_B - H_A + 540) % 360) - 180" in template
        assert "H_mid = (H_A + delta / 2 + 360) % 360" in template
        assert "Verify heading error <=3° before moving." in template
        assert "Compute one-go crossing distance using LiDAR + image geometry" in template
        assert "Do not target either landmark directly for this task." in template
        assert "delta_AB = abs(((H_B - H_A + 540) % 360) - 180)" in template
        assert "D_base = d_front - 0.35" in template
        assert "clamp `D_cross` to `[0.80, 1.80]`" in template

    def test_wall_to_storage_cabinet_rule_exists(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / "mcp"
            / "prompts"
            / "prompt_templates"
            / "assistant_robotics.yaml"
        )
        config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        template = config["instruction_template"]

        assert "Ordered rule for \"move along the wall to storage cabinet\" tasks (strict)" in template
        assert "drive_until_lidar_stop(speed=0.12-0.16, stop_distance_m=0.20, max_distance_m=1.5)" in template
        assert "Approach the wall perpendicularly (normal incidence)" in template
        assert "shortest path to the wall (not diagonal)" in template
        assert "if heading drift exceeds ~5°, stop and re-align" in template
        assert "acceptable 0.18-0.25 m" in template
        assert "about ±90° from H_wall" in template
        assert "move_forward(distance=0.35-0.60, speed=0.16-0.20)" in template
        assert "move_forward(distance=1.00-1.60, speed=0.18-0.22)" in template
        assert "Do NOT declare success just because cabinet is visible." in template
        assert "front clearance >0.8 m" in template
        assert "no closer non-cabinet obstacle is the primary frontal stop trigger" in template
        assert "Do not use a single `get_lidar_scan(summarize=true)` front minimum as sole evidence" in template
        assert "Run heading micro-check around cabinet bearing" in template
        assert "H_cab-10°, H_cab, H_cab+10°" in template
        assert "If the closest return shifts to a side heading" in template

    def test_trash_can_bypass_rule_exists(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / "mcp"
            / "prompts"
            / "prompt_templates"
            / "assistant_robotics.yaml"
        )
        config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        template = config["instruction_template"]

        assert "Ordered rule for \"navigate around trash can blocking aisle\" tasks (strict)" in template
        assert "Turn left 40-50°." in template
        assert "Move forward 0.40-0.50 m to shift to the left side of the can." in template
        assert "Turn right 40-50° to become roughly parallel to original aisle direction." in template
        assert "Move forward 0.40-0.50 m to surpass the can." in template
        assert "Move forward 0.40-0.50 m to re-enter the aisle beyond the can." in template
        assert "Do not declare success immediately after first side-shift; success requires full bypass and re-entry." in template
