"""Tests for imbench.eval.harness.run_benchmark and PolicyInterface."""
from __future__ import annotations

import sys

import numpy as np
import pytest

import imbench as ib
from imbench.eval import harness as harness_mod
from imbench.eval.harness import run_benchmark
from imbench.policy.base_policy import PolicyInterface
from imbench.registry import make


@pytest.fixture
def short_make(monkeypatch):
    """Force every env created inside run_benchmark to truncate in 3 steps.

    run_benchmark calls ``ib.make(task_id, obs_mode=obs_mode, render_mode=...)``
    without a max_episode_steps override, so without this patch the default
    task horizon (often 1000) would make the test multi-minute.
    """
    original = ib.make

    def _short(task_id, **kw):
        kw.setdefault("max_episode_steps", 3)
        return original(task_id, **kw)

    monkeypatch.setattr(harness_mod.ib, "make", _short)
    return _short


class ZeroPolicy(PolicyInterface):
    """Policy that always returns zero actions. Requires action_dim at init."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.reset_count = 0

    def act(self, obs):
        return np.zeros(self.action_dim)

    def reset(self):
        self.reset_count += 1


# ── PolicyInterface contract ────────────────────────────────────────


class TestPolicyInterface:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            PolicyInterface()  # type: ignore[abstract]

    def test_default_reset_is_noop(self):
        # Concrete subclass without overriding reset() still has a no-op reset.
        class P(PolicyInterface):
            def act(self, obs):
                return np.array([0.0])
        p = P()
        assert p.reset() is None

    def test_zero_policy_act(self):
        p = ZeroPolicy(action_dim=7)
        action = p.act({})
        assert isinstance(action, np.ndarray)
        assert action.shape == (7,)
        assert np.all(action == 0.0)


# ── run_benchmark ───────────────────────────────────────────────────


class TestRunBenchmark:

    def test_single_task_single_episode(self, short_make):
        """End-to-end: run a real task for 1 episode with a zero policy."""
        task_id = "ib-p2-cube-toss-v1"
        task = make(task_id, obs_mode="proprio", render_mode=None,
                     seed=0, max_episode_steps=3)
        lo, _ = task.action_space.low, task.action_space.high
        action_dim = lo.shape[0]
        task.close()

        policy = ZeroPolicy(action_dim=action_dim)
        results = run_benchmark(
            policy,
            [task_id],
            n_episodes=1,
            obs_mode="proprio",
            render=False,
        )

        assert task_id in results
        row = results[task_id]
        for key in ("mean_score", "std_score", "success_rate",
                    "mean_length", "n_episodes"):
            assert key in row, f"Missing result key: {key}"
        assert row["n_episodes"] == 1
        assert 0.0 <= row["mean_score"] <= 1.0
        assert 0.0 <= row["success_rate"] <= 1.0
        assert row["mean_length"] >= 1.0
        assert policy.reset_count == 1

    def test_policy_sees_no_ground_truth_by_default(self, short_make):
        class KeyRecorder(ZeroPolicy):
            def act(self, obs):
                self.keys = set(obs)
                return super().act(obs)

        policy = KeyRecorder(action_dim=7)
        run_benchmark(policy, ["ib-p5-occluder-push-v1"], n_episodes=1, obs_mode="proprio")
        assert "robot0_eef_pos" in policy.keys
        assert not policy.keys & {"object_state", "object_pos", "object_quat"}

        run_benchmark(policy, ["ib-p5-occluder-push-v1"], n_episodes=1, obs_mode="proprio",
                      privileged_obs=True)
        assert "object_state" in policy.keys

    def test_multiple_episodes_aggregate(self, short_make):
        task_id = "ib-p2-cube-toss-v1"
        task = make(task_id, obs_mode="proprio", render_mode=None,
                     seed=0, max_episode_steps=3)
        lo, _ = task.action_space.low, task.action_space.high
        action_dim = lo.shape[0]
        task.close()

        policy = ZeroPolicy(action_dim=action_dim)
        results = run_benchmark(
            policy,
            [task_id],
            n_episodes=2,
            seed_offset=100,
            obs_mode="proprio",
            render=False,
        )
        assert results[task_id]["n_episodes"] == 2
        assert policy.reset_count == 2


# ── Metrics on scripted episodes ────────────────────────────────────


class _ScriptedEnv:
    """Stand-in env replaying fixed ``(task_score, is_success, terminated, truncated)`` steps."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.reset_seeds = []
        self.closed = False

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self._steps = iter(self.episodes[len(self.reset_seeds) - 1])
        self._t = 0
        return {}, {}

    def step(self, action):
        score, success, terminated, truncated = next(self._steps)
        self._t += 1
        return {}, score, terminated, truncated, {"task_score": score, "is_success": success, "step": self._t}

    def close(self):
        self.closed = True


class TestHarnessMetrics:
    # Success mid-episode with a low final score; a high score without success;
    # success at termination with a final score below 0.9.
    EPISODES = [
        [(0.2, False, False, False), (0.5, True, False, False), (0.4, False, False, True)],
        [(0.95, False, False, False), (0.95, False, True, False)],
        [(0.8, True, True, False)],
    ]

    @pytest.fixture
    def env(self, monkeypatch):
        env = _ScriptedEnv(self.EPISODES)
        monkeypatch.setattr(harness_mod.ib, "make", lambda task_id, **kw: env)
        return env

    def test_metrics_follow_is_success_and_final_values(self, env):
        policy = ZeroPolicy(7)
        metrics = run_benchmark(policy, ["ib-p2-cube-toss-v1"], n_episodes=3, seed_offset=10)
        m = metrics["ib-p2-cube-toss-v1"]
        finals = [0.4, 0.95, 0.8]
        assert m["success_rate"] == pytest.approx(2 / 3)
        assert m["mean_score"] == pytest.approx(np.mean(finals))
        assert m["std_score"] == pytest.approx(np.std(finals))
        assert m["mean_length"] == pytest.approx(2.0)
        assert m["n_episodes"] == 3
        assert env.reset_seeds == [10, 11, 12]
        assert policy.reset_count == 3
        assert env.closed


# ── Optional seed / env arguments and rendering ─────────────────────


class _SeededOraclePolicy(PolicyInterface):
    """Records the seed each reset receives and the env each act receives."""

    def __init__(self):
        self.seeds = []
        self.envs = []

    def reset(self, seed=None):
        self.seeds.append(seed)

    def act(self, obs, env=None):
        self.envs.append(env)
        return np.zeros(7)


class TestPolicyArguments:

    @pytest.fixture
    def env(self, monkeypatch):
        env = _ScriptedEnv(TestHarnessMetrics.EPISODES)
        monkeypatch.setattr(harness_mod.ib, "make", lambda task_id, **kw: env)
        return env

    def test_seed_passed_and_env_withheld_by_default(self, env):
        policy = _SeededOraclePolicy()
        run_benchmark(policy, ["ib-p2-cube-toss-v1"], n_episodes=3, seed_offset=10)
        assert policy.seeds == [10, 11, 12]
        assert len(policy.envs) == 6 and all(e is None for e in policy.envs)

    def test_env_passed_with_env_access(self, env):
        policy = _SeededOraclePolicy()
        run_benchmark(policy, ["ib-p2-cube-toss-v1"], n_episodes=3, seed_offset=10,
                      env_access=True)
        assert len(policy.envs) == 6 and all(e is env for e in policy.envs)

    def test_act_requiring_env_needs_env_access(self, env):
        class NeedsEnv(PolicyInterface):
            def act(self, obs, env):
                return np.zeros(7)

        with pytest.raises(ValueError, match="env_access"):
            run_benchmark(NeedsEnv(), ["ib-p2-cube-toss-v1"], n_episodes=1)

    def test_env_dependent_oracle_runs(self, monkeypatch):
        # The oracle plans its drop from env state after 8 settle steps; it raised
        # a RuntimeError when the harness called act(obs) only.
        from data_synthetic.single_arm_pipeline.galileo_ramp_drop_policy import (
            GalileoRampDropPolicy,
        )

        original = ib.make
        monkeypatch.setattr(harness_mod.ib, "make",
                            lambda task_id, **kw: original(task_id, max_episode_steps=12, **kw))
        task_id = "ib-p2-galileo-ramp-drop-v1"
        # Scripted oracles read the ball pose from obs, so they need ground truth.
        results = run_benchmark(GalileoRampDropPolicy(), [task_id], n_episodes=1, obs_mode="proprio",
                                privileged_obs=True, env_access=True)
        assert results[task_id]["mean_length"] == 12.0

    def test_render_without_display_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

        def no_make(task_id, **kw):
            raise AssertionError("env created")

        monkeypatch.setattr(harness_mod.ib, "make", no_make)
        with pytest.raises(RuntimeError, match="DISPLAY"):
            run_benchmark(ZeroPolicy(7), ["ib-p2-cube-toss-v1"], n_episodes=1, render=True)
