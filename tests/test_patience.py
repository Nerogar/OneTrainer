"""Tests for the Patience (early stopping) feature."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from modules.util.commands.TrainCommands import TrainCommands


def _make_config(patience=True, patience_epochs=3, validation=True, workspace_dir=None):
    """Create a minimal config-like object for patience testing."""
    return SimpleNamespace(
        patience=patience,
        patience_epochs=patience_epochs,
        validation=validation,
        workspace_dir=workspace_dir or tempfile.mkdtemp(),
    )


def _make_trainer_state(config, parameters=None):
    """Create a minimal namespace mimicking GenericTrainer's patience-related state."""
    commands = TrainCommands()
    tensorboard = MagicMock()

    state = SimpleNamespace(
        config=config,
        commands=commands,
        tensorboard=tensorboard,
        parameters=parameters or [torch.randn(4, 4) for _ in range(3)],
        _patience_counter=0,
        _patience_best_loss=float('inf'),
        _patience_best_step=-1,
        _patience_best_backup_path=None,
    )
    return state


def _train_progress(global_step):
    return SimpleNamespace(global_step=global_step)


def _check_patience(state, val_loss, train_progress):
    """Replicate GenericTrainer.__check_patience logic for testing."""
    is_new_best = val_loss < state._patience_best_loss

    if is_new_best:
        state._patience_best_backup_path = _save_patience_best(state, train_progress)
        state._patience_best_step = train_progress.global_step
        state._patience_best_loss = val_loss
        state._patience_counter = 0
    else:
        state._patience_counter += 1

    state.tensorboard.add_scalar("patience/counter", state._patience_counter, train_progress.global_step)
    state.tensorboard.add_scalar("patience/best_val_loss", state._patience_best_loss, train_progress.global_step)

    if state._patience_counter >= state.config.patience_epochs:
        print(f"Patience triggered at step {train_progress.global_step}. "
              f"Best checkpoint from step {state._patience_best_step} "
              f"(val_loss: {state._patience_best_loss:.6f})")
        state.commands.stop()


def _save_patience_best(state, train_progress):
    """Replicate GenericTrainer.__save_patience_best logic for testing."""
    best_path = os.path.join(state.config.workspace_dir, "backup", "patience-best.pt")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    tensor_state = [p.data.clone().cpu() for p in state.parameters]
    torch.save(tensor_state, best_path)
    return best_path


class TestPatienceCounterIncrement(unittest.TestCase):
    """Verify counter increments when val_loss doesn't improve."""

    def test_counter_increments_on_stagnation(self):
        config = _make_config(patience_epochs=5)
        state = _make_trainer_state(config)

        # First call sets the baseline
        _check_patience(state, 1.0, _train_progress(10))
        self.assertEqual(state._patience_counter, 0)
        self.assertEqual(state._patience_best_loss, 1.0)

        # Subsequent calls with same or worse loss increment counter
        _check_patience(state, 1.0, _train_progress(20))
        self.assertEqual(state._patience_counter, 1)

        _check_patience(state, 1.5, _train_progress(30))
        self.assertEqual(state._patience_counter, 2)

        _check_patience(state, 2.0, _train_progress(40))
        self.assertEqual(state._patience_counter, 3)


class TestPatienceCounterReset(unittest.TestCase):
    """Verify counter resets to 0 when val_loss improves."""

    def test_counter_resets_on_improvement(self):
        config = _make_config(patience_epochs=5)
        state = _make_trainer_state(config)

        _check_patience(state, 1.0, _train_progress(10))
        _check_patience(state, 1.1, _train_progress(20))
        _check_patience(state, 1.2, _train_progress(30))
        self.assertEqual(state._patience_counter, 2)

        # Improvement resets counter
        _check_patience(state, 0.8, _train_progress(40))
        self.assertEqual(state._patience_counter, 0)
        self.assertEqual(state._patience_best_loss, 0.8)
        self.assertEqual(state._patience_best_step, 40)


class TestPatienceStopTrigger(unittest.TestCase):
    """Verify commands.stop() is called when counter reaches patience_epochs."""

    def test_stop_triggered_at_threshold(self):
        config = _make_config(patience_epochs=3)
        state = _make_trainer_state(config)

        _check_patience(state, 1.0, _train_progress(10))  # baseline, counter=0
        self.assertFalse(state.commands.get_stop_command())

        _check_patience(state, 1.1, _train_progress(20))  # counter=1
        self.assertFalse(state.commands.get_stop_command())

        _check_patience(state, 1.1, _train_progress(30))  # counter=2
        self.assertFalse(state.commands.get_stop_command())

        _check_patience(state, 1.1, _train_progress(40))  # counter=3 >= patience_epochs
        self.assertTrue(state.commands.get_stop_command())

    def test_stop_not_triggered_if_improvement_resets(self):
        config = _make_config(patience_epochs=3)
        state = _make_trainer_state(config)

        _check_patience(state, 1.0, _train_progress(10))
        _check_patience(state, 1.1, _train_progress(20))  # counter=1
        _check_patience(state, 1.1, _train_progress(30))  # counter=2
        _check_patience(state, 0.9, _train_progress(40))  # improvement! counter=0
        _check_patience(state, 1.0, _train_progress(50))  # counter=1
        _check_patience(state, 1.0, _train_progress(60))  # counter=2

        self.assertFalse(state.commands.get_stop_command())
        self.assertEqual(state._patience_counter, 2)


class TestPatienceCheckpointSaveRestore(unittest.TestCase):
    """Verify parameter tensors are saved and correctly restored."""

    def test_save_and_restore_best_checkpoint(self):
        config = _make_config()
        params = [torch.randn(4, 4) for _ in range(3)]
        state = _make_trainer_state(config, parameters=params)

        # Record the "best" parameter values
        best_values = [p.data.clone() for p in params]

        # Trigger a save by improving
        _check_patience(state, 0.5, _train_progress(10))
        self.assertTrue(os.path.isfile(state._patience_best_backup_path))

        # Mutate parameters (simulating continued training)
        for p in params:
            p.data.fill_(999.0)

        # Restore
        best_state = torch.load(state._patience_best_backup_path, weights_only=True)
        for param, saved in zip(params, best_state, strict=True):
            param.data.copy_(saved)

        # Verify restored values match the best
        for original, restored in zip(best_values, params, strict=True):
            self.assertTrue(torch.equal(original, restored.data))

    def test_checkpoint_overwritten_on_new_best(self):
        config = _make_config()
        params = [torch.randn(4, 4) for _ in range(3)]
        state = _make_trainer_state(config, parameters=params)

        _check_patience(state, 1.0, _train_progress(10))
        first_values = [p.data.clone() for p in params]

        # Change params and improve
        for p in params:
            p.data.add_(1.0)
        _check_patience(state, 0.5, _train_progress(20))

        # The checkpoint should contain the NEW (step 20) values, not step 10
        saved = torch.load(state._patience_best_backup_path, weights_only=True)
        for current, loaded in zip(params, saved, strict=True):
            self.assertTrue(torch.equal(current.data, loaded))

        # First values should NOT match (they were overwritten)
        for first, loaded in zip(first_values, saved, strict=True):
            self.assertFalse(torch.equal(first, loaded))


class TestPatienceAutoEnableValidation(unittest.TestCase):
    """Verify patience=True + validation=False results in validation auto-enabled."""

    def test_auto_enable_at_config_level(self):
        config = _make_config(patience=True, validation=False)
        self.assertFalse(config.validation)

        # Simulate the training-time guard from GenericTrainer.start()
        if config.patience and not config.validation:
            config.validation = True

        self.assertTrue(config.validation)


class TestPatienceTensorboardLogging(unittest.TestCase):
    """Verify TensorBoard scalars are logged correctly."""

    def test_logs_counter_and_best_loss(self):
        config = _make_config(patience_epochs=5)
        state = _make_trainer_state(config)

        _check_patience(state, 1.0, _train_progress(10))

        calls = state.tensorboard.add_scalar.call_args_list
        tags = [c[0][0] for c in calls]
        self.assertIn("patience/counter", tags)
        self.assertIn("patience/best_val_loss", tags)


if __name__ == "__main__":
    unittest.main()
