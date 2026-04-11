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

        assert "motion target is the connecting line and its ingress point" in template
        assert "Prefer inference while moving for dynamic scenes" in template
        assert "Compute the heading in between the two landmarks" in template
        assert "delta = ((H_B - H_A + 540) % 360) - 180" in template
        assert "H_mid = (H_A + delta / 2 + 360) % 360" in template
        assert "Verify heading error <=3° before moving." in template
        assert "Turn to that in-between heading" in template
        assert "x = d_front" in template
        assert "D_pass = 1.2 * x" in template
        assert "D_extra = max(0.60, 0.50 * x)" in template
        assert "No hard clamp for this task" in template
        assert "continue forward in additional 0.30-0.60 m segments" in template
        assert "both landmarks are behind the robot (rear hemisphere confirmation on re-scan)" in template
        assert "angdiff(H_obj, H_mid) >= 90°" in template
        assert "Do NOT output planned actions without executing them." in template

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

        assert 'Ordered rule for "move along the wall to storage cabinet" tasks (strict)' in template
        assert "Hard execution order (mandatory)" in template
        assert "Wall-first is mandatory." in template
        assert "Do NOT scan for cabinet, turn to cabinet heading," in template
        assert "or approach cabinet before wall setup is complete." in template
        assert "acceptable band 0.18-0.22 m" in template
        assert "Find nearest wall and approach it" in template
        assert "drive_until_lidar_stop(stop_distance_m=0.20" in template
        assert "Stop when front lidar is about 0.20 m from the wall" in template
        assert "Too-near recovery (required)" in template
        assert "turn(angle=180)" in template
        assert "move_forward(distance=0.30)" in template
        assert "Rotate and find cabinet" in template
        assert "turn_to_heading(target_heading_deg=H_cabinet)" in template
        assert "Approach cabinet to 0.20-0.30 m" in template
        assert "Final stop band is 0.20-0.30 m from cabinet." in template
        assert "follow_wall_lidar" not in template

    def test_patrol_between_lab_chairs_rule_exists(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / "mcp"
            / "prompts"
            / "prompt_templates"
            / "assistant_robotics.yaml"
        )
        config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        template = config["instruction_template"]

        assert 'Ordered rule for "patrol between the two lab chairs" tasks (strict)' in template
        assert "Find two chairs with vision-first search" in template
        assert "Treat LiDAR as safety-only (chair legs are often sparse/noisy)." in template
        assert "Patrol 2-3 cycles" in template
        assert "Approach each chair to ~0.30-0.45 m" in template
        assert "If LiDAR triggers but chair is off-center, re-align to chair visually and continue in shorter steps." in template
        assert "Complete after 2 full A↔B cycles; optionally do a 3rd" in template

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
        assert "stop_distance_m=0.30" in template
        assert "Do not start bypass if already too close (<0.30 m)" in template
        assert "Use camera as the primary sensor for initial target identification" in template
        assert "Sensor handoff (required): once near the trash can (about <=0.8 m)" in template
        assert "switch to LiDAR-dominant local navigation" in template
        assert "Perform square movement around the can (counterclockwise, required)" in template
        assert "Leg 1 (left offset edge): turn left 90°, then move forward 0.30 m." in template
        assert "Leg 2: turn left 90°, then move forward 0.60 m." in template
        assert "Leg 3: turn left 90°, then move forward 0.60 m." in template
        assert "Leg 4: turn left 90°, then move forward 0.60 m." in template
        assert "fixed square-style bypass pattern" in template
        assert "min_distance_m >= 0.25 m" in template
        assert "prioritize LiDAR clearance decisions over camera-only proximity impressions" in template
        assert "Final realignment to aisle (required)" in template
        assert "Do not run completion checks until this final aisle alignment is done." in template
        assert "Completion criteria (required, motion-separated)" in template
        assert "Move forward a short confirmation segment (about 0.20-0.30 m)" in template
        assert "post-motion re-check confirmation" in template
