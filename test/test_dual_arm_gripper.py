"""Offline gripper regression tests; all hardware entry points are mocked."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from test import dual_arm_control as control


def fake_gripper(width=0.08):
    gripper = MagicMock()
    state = SimpleNamespace(width=width, max_width=0.08, is_grasped=False)
    gripper.read_once.return_value = state
    gripper.homing.return_value = True

    def move(target, speed):
        state.width = target
        return True

    def grasp(target, *args):
        state.width = target
        state.is_grasped = True
        return True

    gripper.move.side_effect = move
    gripper.grasp.side_effect = grasp
    return gripper


class GripperTests(unittest.TestCase):
    def test_initial_width_is_commanded(self):
        gripper = fake_gripper(0.04)
        with patch.object(control.franka, "Gripper", return_value=gripper):
            actual, width = control._prepare_gripper("fake", "Left", [{"left_gripper": 0.08}])
        self.assertIs(actual, gripper)
        self.assertEqual(width, 0.08)
        gripper.move.assert_called_once_with(0.08, control.kGripperMoveSpeed)

    def test_failed_connection_or_homing_aborts(self):
        with patch.object(control.franka, "Gripper", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "Left gripper setup failed.*offline"):
                control._prepare_gripper("fake", "Left", [{"left_gripper": 0.08}])
        gripper = fake_gripper()
        gripper.homing.return_value = False
        with patch.object(control.franka, "Gripper", return_value=gripper):
            with self.assertRaisesRegex(RuntimeError, "homing returned False"):
                control._prepare_gripper("fake", "Left", [{"left_gripper": 0.08}])
        gripper.move.assert_not_called()

    def test_unreachable_opening_clamps_to_max(self):
        gripper = fake_gripper()
        with patch.object(control.franka, "Gripper", return_value=gripper):
            actual, width = control._prepare_gripper("fake", "Left", [{"left_gripper": 0.09}])
        self.assertIs(actual, gripper)
        self.assertEqual(width, 0.08)
        gripper.move.assert_called_once_with(0.08, control.kGripperMoveSpeed)

    def test_failed_initial_move_aborts(self):
        gripper = fake_gripper()
        gripper.move.side_effect = None
        gripper.move.return_value = False
        with patch.object(control.franka, "Gripper", return_value=gripper):
            with self.assertRaisesRegex(RuntimeError, "initial move.*returned False"):
                control._prepare_gripper("fake", "Left", [{"left_gripper": 0.08}])

    def test_missing_initial_width_uses_measured_width(self):
        gripper = fake_gripper()
        with patch.object(control.franka, "Gripper", return_value=gripper):
            _, width = control._prepare_gripper("fake", "Left", [{}, {"left_gripper": 0.055}])
        self.assertEqual(width, 0.08)
        gripper.move.assert_not_called()

    def test_grasp_with_no_object_succeeds(self):
        gripper = fake_gripper()
        gripper.grasp.side_effect = None
        gripper.grasp.return_value = False  # No object detected
        errors = []
        control._command_gripper_worker(gripper, 0.040, 0.08, "Left", errors)
        # Grasp completing without finding an object is not an error
        self.assertEqual(errors, [])
        gripper.grasp.assert_called_once_with(
            0.040,
            control.kGripperMoveSpeed,
            control.kGripperForce,
        )

    def test_handover_dispatches_all_eight_actions(self):
        path = Path(__file__).resolve().parents[1] / "trajectories/handover.json"
        trajectory = json.loads(path.read_text())
        grippers = [fake_gripper(), fake_gripper()]
        controllers = [MagicMock(), MagicMock()]
        for controller in controllers:
            controller.get_target_feedback.return_value.error_message = None
            controller.set_joint_position_target.return_value.error_message = None
        actions = []

        def dispatch(gripper, width, previous, label, workers, errors):
            actions.append((label, "grasp" if width < previous else "place"))
            control._command_gripper_worker(gripper, width, previous, label, errors)

        def sleep(seconds):
            if seconds == 0.1:  # Exit the final user-wait loop.
                control.motion_finished = True

        with (
            patch.object(control.franka, "Robot"),
            patch.object(control.franka, "Gripper", side_effect=grippers),
            patch.object(control, "configure_position_controller", side_effect=controllers),
            patch.object(control, "setDefaultBehaviour"),
            patch.object(control, "move_robots_to_start_pose"),
            patch.object(control, "_start_gripper_command", side_effect=dispatch),
            patch.object(control.signal, "signal"),
            patch.object(control.time, "sleep", side_effect=sleep),
            patch.object(control.time, "monotonic", return_value=0.0),
            patch.object(control, "motion_finished", False),
        ):
            control.run_hardware_execution(str(path))
        self.assertEqual(actions, [
            ("Left", "grasp"), ("Right", "grasp"), ("Left", "place"),
            ("Left", "grasp"), ("Right", "place"), ("Right", "grasp"),
            ("Left", "place"), ("Right", "place"),
        ])
        for controller in controllers:
            self.assertEqual(controller.set_joint_position_target.call_count, len(trajectory))
            controller.stop_control.assert_called_once()


if __name__ == "__main__":
    unittest.main()
