"""Gymnasium compatibility test for IMBench.

Tests performed
---------------
1. Import check: imbench registers itself with gymnasium on import.
2. gymnasium.make() — create an env via the gymnasium registry.
3. Spaces — observation_space is a Dict, action_space a bounded Box.
4. API smoke test — reset / step / render / close cycle.
5. Seeded determinism — two resets with the same seed produce identical obs.
6. Random-policy rollout — 3 episodes of up to 50 steps with action_space.sample().
7. env_checker — run gymnasium's check_env on the env.
8. SB3 PPO — if stable-baselines3 is installed, train for a short burst and
   confirm that a policy can be learned without crashes; skipped otherwise.

Usage
-----
    python scripts/test_gymnasium_compat.py

The script exits with code 0 when no check fails (skips do not count), 1 otherwise.
"""

from __future__ import annotations

import sys
import traceback
import numpy as np

# ── which task to test ──────────────────────────────────────────────────────
# Pick a single-arm task with no cameras for speed; override obs_mode to
# "proprio" so the test doesn't spin up offscreen GL rendering.
TASK_ID    = "ib-p1-edge-slide-v1"
TASK_KWARGS = dict(obs_mode="proprio", seed=0)


# ════════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"

_results: list[tuple[str, bool, str]] = []
_skipped: list[tuple[str, str]] = []


def record(name: str, passed: bool, note: str = "") -> None:
    _results.append((name, passed, note))
    tag = PASS if passed else FAIL
    msg = f"  {note}" if note else ""
    print(f"{tag} {name}{msg}")


def skip(name: str, note: str) -> None:
    _skipped.append((name, note))
    print(f"{SKIP} {name}  {note}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ════════════════════════════════════════════════════════════════════════════
# 1. import + registration
# ════════════════════════════════════════════════════════════════════════════
section("1. Import & registration")

try:
    import gymnasium
    import imbench  # noqa — triggers gymnasium.register() side-effect
    registered = TASK_ID in gymnasium.envs.registry
    record("gymnasium import", True)
    record(f"'{TASK_ID}' in gymnasium.envs.registry", registered)
except Exception as e:
    record("import imbench", False, str(e))
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# 2. gymnasium.make()
# ════════════════════════════════════════════════════════════════════════════
section("2. gymnasium.make()")

env = None
try:
    env = gymnasium.make(TASK_ID, disable_env_checker=True, **TASK_KWARGS)
    record("gymnasium.make()", True, f"type={type(env).__name__}")
except Exception as e:
    record("gymnasium.make()", False, str(e))
    traceback.print_exc()
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# 3. Spaces validation
# ════════════════════════════════════════════════════════════════════════════
section("3. Spaces validation")

try:
    obs_space = env.observation_space
    act_space = env.action_space
    record("observation_space is Dict",  isinstance(obs_space, gymnasium.spaces.Dict))
    record("action_space is Box",        isinstance(act_space, gymnasium.spaces.Box))
    record("action_space bounded",       np.all(np.isfinite(act_space.low)) and
                                         np.all(np.isfinite(act_space.high)))
    print(f"    obs keys : {list(obs_space.spaces.keys())}")
    print(f"    action dim: {act_space.shape}")
except Exception as e:
    record("spaces validation", False, str(e))
    traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# 4. API smoke test: reset / step / render / close
# ════════════════════════════════════════════════════════════════════════════
section("4. API smoke test")

try:
    obs, info = env.reset(seed=42)
    record("reset() returns (obs, info)",     isinstance(obs, dict) and isinstance(info, dict))
    record("obs in observation_space",        env.observation_space.contains(obs))

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    record("step() returns 5-tuple",          True)
    record("reward is float",                 isinstance(reward, (float, np.floating)))
    record("terminated is bool",              isinstance(terminated, bool))
    record("truncated is bool",               isinstance(truncated, bool))
    record("step obs in observation_space",   env.observation_space.contains(obs2))
    record("'task_score' in info",            "task_score" in info2)

    # render — rgb_array not applicable here (no cameras), but shouldn't crash
    try:
        frame = env.render()
        record("render() doesn't crash", True)
    except Exception as e:
        record("render() doesn't crash", False, str(e))

except Exception as e:
    record("API smoke test", False, str(e))
    traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# 5. Seeded determinism
# ════════════════════════════════════════════════════════════════════════════
section("5. Seeded determinism")

try:
    obs_a, _ = env.reset(seed=7)
    obs_b, _ = env.reset(seed=7)
    keys_match = set(obs_a) == set(obs_b)
    vals_match  = all(np.allclose(obs_a[k], obs_b[k]) for k in obs_a)
    record("same seed → same initial obs", keys_match and vals_match)
except Exception as e:
    record("seeded determinism", False, str(e))
    traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# 6. Random-policy rollout (3 episodes, up to 50 steps each)
# ════════════════════════════════════════════════════════════════════════════
section("6. Random-policy rollout (3 eps × 50 steps)")

try:
    episodes_ok = 0
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        done = False
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                break
        episodes_ok += 1
        print(f"    ep {ep}: {'done early' if done else '50 steps'}, "
              f"score={info.get('task_score', '?'):.3f}")
    record("3 random-policy episodes completed", episodes_ok == 3)
except Exception as e:
    record("random rollout", False, str(e))
    traceback.print_exc()

env.close()


# ════════════════════════════════════════════════════════════════════════════
# 7. gymnasium env_checker
# ════════════════════════════════════════════════════════════════════════════
section("7. gymnasium env_checker (PassiveEnvChecker)")

try:
    from gymnasium.utils.env_checker import check_env

    # Re-create a fresh env for the checker so state is clean
    check_env_inst = gymnasium.make(TASK_ID, disable_env_checker=True, **TASK_KWARGS)
    # check_env will call reset + a few steps and validate everything
    check_env(check_env_inst.unwrapped, warn=True, skip_render_check=True)
    check_env_inst.close()
    record("gymnasium.utils.env_checker.check_env()", True)
except AssertionError as e:
    record("gymnasium.utils.env_checker.check_env()", False, str(e))
    traceback.print_exc()
except Exception as e:
    record("gymnasium.utils.env_checker.check_env()", False, str(e))
    traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# 8. Stable-Baselines3 PPO (short training burst)
# ════════════════════════════════════════════════════════════════════════════
section("8. SB3 PPO integration (2 048 timesteps)")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    # Optional: stable-baselines3 (and torch) are in no IMBench extra.
    skip("SB3 PPO training", f"{e} — install with: pip install stable-baselines3")
else:
    try:
        # SB3's PPO requires a flat observation space; we flatten with FlattenObservation
        from gymnasium.wrappers import FlattenObservation

        def _make_flat():
            return FlattenObservation(
                gymnasium.make(TASK_ID, disable_env_checker=True, **TASK_KWARGS)
            )

        flat_vec = DummyVecEnv([_make_flat])
        model = PPO("MlpPolicy", flat_vec, verbose=0, n_steps=64, batch_size=32)
        model.learn(total_timesteps=2048)
        flat_vec.close()
        record("SB3 PPO 2048-step training", True)
    except Exception as e:
        record("SB3 PPO training", False, str(e))
        traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
section("Summary")

passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
print(f"  {passed} passed, {failed} failed, {len(_skipped)} skipped\n")

for name, ok, note in _results:
    tag = PASS if ok else FAIL
    msg = f"  ({note})" if note and not ok else ""
    print(f"  {tag} {name}{msg}")
for name, note in _skipped:
    print(f"  {SKIP} {name}  ({note})")

sys.exit(0 if failed == 0 else 1)
