"""Regression tests for scoring/termination/reset exploits in the bimanual tasks.

Covers the heterogeneous (``hetbi``) and homogeneous (``hobi``) task
families: partial credit must stay below 1 without success, success must
hold all of its conditions at once, the payload's start must be valid and
reproducible, and the reset frame must show the final scene.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import yaml

import imbench as ib
from imbench.tasks.base_task import CONFIGS_DIR

_HETBI_IDS = sorted(p.stem for p in (CONFIGS_DIR / "hetbi").glob("*.yaml"))
_PLACEMENT_IDS = _HETBI_IDS + ["ib-p1-plate-shelf-hobi-v1"]


def _zero(task):
    return np.zeros(task.action_space.shape)


def _step_n(task, n):
    for _ in range(n):
        step = task.step(_zero(task))
        if step[2] or step[3]:
            break
    return step


def _set_free_joint(task, joint, pos, quat_wxyz):
    sim = task._env.sim
    sim.data.set_joint_qpos(joint, np.concatenate([pos, quat_wxyz]))
    sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()


def _quat_about(axis: str, deg: float) -> np.ndarray:
    """wxyz quaternion for a rotation of *deg* about world *axis*."""
    half = np.radians(deg) / 2.0
    q = np.zeros(4)
    q[0] = np.cos(half)
    q["xyz".index(axis) + 1] = np.sin(half)
    return q


# =====================================================================
#  BI-01 / BI-06: no free or sticky credit reaching 1 without success
# =====================================================================

@pytest.mark.parametrize("task_id", _PLACEMENT_IDS)
def test_no_free_credit_at_rest(task_id):
    """A zero-action policy earns (almost) nothing: no stage is satisfied at spawn."""
    task = ib.make(task_id, obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        _, _, terminated, _, info = _step_n(task, 30)
        assert info["task_score"] < 0.02, task._stage_debug
        assert info["is_success"] is False
        assert terminated is False
    finally:
        task.close()


@pytest.mark.parametrize("task_id", [
    "ib-p1-bowl-table-v1", "ib-p1-cup-inversion-v1",
    "ib-p1-plate-shelf-hetbi-v1", "ib-p1-plate-shelf-hobi-v1",
])
def test_final_stage_is_the_success_condition(task_id):
    """Sticky stages alone cannot sum to 1: the last stage tracks success live."""
    task = ib.make(task_id, obs_mode="proprio", seed=0)
    try:
        final = task._staged_reward._stages[-1]
        assert final.sticky is False
        assert sum(s.weight for s in task._staged_reward._stages[:-1]) <= 0.9 + 1e-9
    finally:
        task.close()


def test_cup_sticky_milestones_do_not_succeed():
    """Touching each milestone at a different time never
    produces success, termination or a score of 1."""
    task = ib.make("ib-p1-cup-inversion-v1", obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        sim = task._env.sim
        joint = task._env._payload.joints[0]
        start = np.array(sim.data.get_joint_qpos(joint))
        snout = sim.data.geom_xpos[sim.model.geom_name2id("gripper1_right_geo_snout")].copy()
        pad = sim.data.geom_xpos[sim.model.geom_name2id("gripper0_right_finger1_pad_collision")].copy()
        goal = task._goal_pos
        for pos, quat in [
            (snout, start[3:]),
            (pad, start[3:]),
            (np.array([goal[0], goal[1], task._table_height + 0.25]), np.array([1.0, 0, 0, 0])),
            (start[:3], start[3:]),
        ]:
            _set_free_joint(task, joint, pos, quat)
            assert task.task_score(recompute=True) < 1.0
        _, _, terminated, _, info = _step_n(task, 30)
        assert terminated is False
        assert info["is_success"] is False
        assert info["task_score"] <= 0.9
    finally:
        task.close()


def test_hobi_plate_flat_in_shelf_not_inserted():
    """A plate lying flat inside the shelf bounds earns no insertion credit."""
    task = ib.make("ib-p1-plate-shelf-hobi-v1", obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        fb = task._fixture_bounds
        pos = np.array([np.mean(fb["x"]), np.mean(fb["y"]), fb["z"][0] + 0.02])
        _set_free_joint(task, task._plate.joints[0], pos, np.array([1.0, 0, 0, 0]))
        _, _, terminated, _, info = _step_n(task, 20)
        assert task._stage_debug["inserted_in_bounds"] == 0.0
        assert terminated is False
        assert info["is_success"] is False
    finally:
        task.close()


def test_hobi_plate_success_terminates():
    """A plate standing on edge in the shelf's centre slot settles into success."""
    task = ib.make("ib-p1-plate-shelf-hobi-v1", obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        fb = task._fixture_bounds
        z = fb["z"][0] + task._plate.horizontal_radius + 0.003
        pos = np.array([np.mean(fb["x"]), np.mean(fb["y"]), z])
        _set_free_joint(task, task._plate.joints[0], pos, _quat_about("x", 90))
        _, _, terminated, _, info = _step_n(task, 80)
        assert terminated is True
        assert info["is_success"] is True
        assert info["task_score"] == 1.0
    finally:
        task.close()


def test_hobi_plate_rewards_one_hand_not_two(monkeypatch):
    """The paper's plan is press-then-grasp with one carrying arm: a plate held
    up by both grippers earns nothing, the same plate held by one earns lift credit."""
    task = ib.make("ib-p1-plate-shelf-hobi-v1", obs_mode="proprio", seed=0)
    try:
        for holders in ({0, 1}, {0}):
            task.reset(seed=0)
            joint = task._plate.joints[0]
            start = np.array(task._env.sim.data.get_joint_qpos(joint))
            _set_free_joint(task, joint, start[:3] + [0.0, 0.0, 0.10], _quat_about("x", 20))
            monkeypatch.setattr(task, "_gripper_contacts_plate", lambda arm: arm in holders)
            monkeypatch.setattr(task, "_gripper_touches_plate", lambda arm: arm in holders)
            task.task_score(recompute=True)
            earned = {k for k, v in task._stage_debug.items() if k != "_success" and v > 0}
            if len(holders) == 2:
                assert not earned, task._stage_debug
            else:
                assert {"edge_grasp", "single_arm_lift"} <= earned, task._stage_debug
    finally:
        task.close()


def test_hetbi_at_rest_needs_a_still_released_window():
    """A payload released mid-air is not at rest until it has been still for the whole window."""
    task = ib.make("ib-p1-cup-inversion-v1", obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        sim = task._env.sim
        joint = task._env._payload.joints[0]
        qpos = np.array(sim.data.get_joint_qpos(joint))
        _set_free_joint(task, joint, qpos[:3] + [0.0, 0.0, 0.015], qpos[3:])
        n = round(0.25 * task.control_freq)  # the 0.25 s at-rest window
        stable = []
        for _ in range(n + 60):
            task.step(_zero(task))
            stable.append(task._payload_is_stable_multistep())
        assert not any(stable[:n])
        assert all(stable[-20:])
    finally:
        task.close()


def test_success_flag_not_stale_after_fall():
    """A fall right after a successful episode must not report success."""
    task = ib.make("ib-p1-cup-inversion-v1", obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        task._stage_debug = {"_success": True}
        task.reset(seed=0)
        assert task._stage_debug["_success"] is False
        _set_free_joint(task, task._env._payload.joints[0], np.array([0.0, 0.0, 0.3]),
                        np.array([1.0, 0, 0, 0]))
        _, _, terminated, _, info = task.step(_zero(task))
        assert terminated is True
        assert info["is_success"] is False
    finally:
        task.close()


# =====================================================================
#  BI-02: seeded resets reproduce, unseeded resets vary
# =====================================================================

def _scene_fingerprint(task) -> str:
    sim = task._env.sim
    parts = [sim.data.qpos, sim.model.light_pos.ravel(), sim.model.body_pos.ravel()]
    blob = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts]).round(9).tobytes()
    goal_xy = getattr(task, "_goal_xy", None)
    if goal_xy:
        blob += repr(sorted((k, tuple(np.round(v, 9))) for k, v in goal_xy.items())).encode()
    return hashlib.md5(blob).hexdigest()


@pytest.mark.parametrize("task_id", [
    "ib-p1-plate-shelf-hobi-v1", "ib-p1-cup-inversion-v1", "ib-m-dining-table-v1",
])
def test_seeded_resets_reproduce_unseeded_vary(task_id):
    task = ib.make(task_id, obs_mode="proprio", seed=0)
    try:
        task.reset(seed=7)
        a = _scene_fingerprint(task)
        task.reset(seed=7)
        assert _scene_fingerprint(task) == a
        task.reset()
        b = _scene_fingerprint(task)
        task.reset()
        c = _scene_fingerprint(task)
        assert b != c
        task.reset(seed=8)
        assert _scene_fingerprint(task) != a
    finally:
        task.close()


# =====================================================================
#  BI-03: dining placement is live, orientation-aware and at rest
# =====================================================================

@pytest.fixture(scope="module")
def dining():
    task = ib.make("ib-m-dining-table-v1", obs_mode="proprio", seed=0)
    yield task
    task.close()


def _dining_teleport(task, name, pos, quat_wxyz):
    _set_free_joint(task, task._ycb_objects[name].joints[0], pos, quat_wxyz)


def _dining_rest_pos(task, name, dz=0.002):
    table_z = float(task._env.table_offset[2])
    gxy = task._goal_xy[name]
    return np.array([gxy[0], gxy[1], task._object_rest_z(name, table_z) + dz])


@pytest.mark.parametrize("seed", [1, 7, 13, 25])
def test_dining_spawns_clear_both_home_grippers(dining, seed):
    """No object spawns in contact with either robot at its home pose."""
    from imbench.tasks.prehensile.contacts import body_geom_ids, robot_contacts

    dining.reset(seed=seed)
    sim = dining._env.sim
    sim.forward()
    for name, obj in dining._ycb_objects.items():
        assert not robot_contacts(sim, body_geom_ids(sim, obj.root_body)), name


def test_dining_grasp_without_real_lift_earns_no_transport(dining, monkeypatch):
    """A grasped object raised a few mm (closing on it, dragging it) earns no transport credit."""
    dining.reset(seed=0)
    name = dining._arm1_objects[0]
    monkeypatch.setattr(dining, "_gripper_contacts_ycb", lambda _name: True)
    table_z = float(dining._env.table_offset[2])
    rest_z = dining._object_rest_z(name, table_z)
    gxy = dining._goal_xy[name]
    upright = np.array([1.0, 0, 0, 0])
    _dining_teleport(dining, name, [gxy[0], gxy[1], rest_z + 0.005], upright)
    dining._lift_progress(name, {})
    assert dining._transport_progress(name, {}) == 0.0
    _dining_teleport(dining, name, [gxy[0], gxy[1], rest_z + dining._MIN_LIFT + 0.005], upright)
    dining._lift_progress(name, {})
    assert dining._transport_progress(name, {}) > 0.9


def test_dining_tipped_objects_at_goals_not_success(dining):
    """Every object lying on its side on its goal: no placement, no success."""
    dining.reset(seed=1)
    table_z = float(dining._env.table_offset[2])
    names = dining._arm0_objects + dining._arm1_objects
    for name in names:
        gxy = dining._goal_xy[name]
        _dining_teleport(dining, name, [gxy[0], gxy[1], table_z + 0.05], _quat_about("x", 90))
    _, _, terminated, _, info = _step_n(dining, 40)
    assert terminated is False
    assert info["is_success"] is False
    for name in names:
        if name not in dining._tilt_exempt:
            assert info["placed"][name] is False, name


def test_dining_placement_requires_rest_and_is_not_latched(dining):
    dining.reset(seed=0)
    name = "tomato_soup_can"
    _dining_teleport(dining, name, _dining_rest_pos(dining, name), np.array([1.0, 0, 0, 0]))
    dining.step(_zero(dining))
    assert dining._placed[name] is False, "placed before being at rest for rest_steps"
    _, _, _, _, info = _step_n(dining, dining._rest_steps + 5)
    assert info["placed"][name] is True
    _dining_teleport(dining, name, np.array([0.0, 0.30, _dining_rest_pos(dining, name)[2]]),
                     np.array([1.0, 0, 0, 0]))
    _, _, _, _, info = dining.step(_zero(dining))
    assert info["placed"][name] is False
    assert info["placed_count"] == sum(info["placed"].values())


def test_dining_placement_has_lower_z_bound(dining):
    dining.reset(seed=0)
    name = "tomato_soup_can"
    table_z = float(dining._env.table_offset[2])
    pos = _dining_rest_pos(dining, name, dz=-(dining._z_tolerance + 0.02))
    _dining_teleport(dining, name, pos, np.array([1.0, 0, 0, 0]))
    assert dining._object_on_goal(name, table_z) is False


def test_dining_all_upright_at_goals_succeeds(dining):
    dining.reset(seed=0)
    names = dining._arm0_objects + dining._arm1_objects
    for name in names:
        _dining_teleport(dining, name, _dining_rest_pos(dining, name), np.array([1.0, 0, 0, 0]))
    _, _, terminated, _, info = _step_n(dining, dining._rest_steps + 30)
    assert info["placed"] == {name: True for name in names}
    assert info["is_success"] is True
    assert terminated is True
    assert info["task_score"] == 1.0


def test_dining_info_keys(dining):
    dining.reset(seed=0)
    _, _, _, _, info = dining.step(_zero(dining))
    for key in ("task_score", "is_success", "step", "placed_count", "total_objects", "placed"):
        assert key in info, key


# =====================================================================
#  BI-04 / BI-05: cracker-box slot insertion
# =====================================================================

def test_jaw_variant_slot_sits_on_table():
    """The jaw variant's slot rests on the table top like the slot variant's,
    both from its YAML pose and from the rotation-aware default height."""
    for overrides in (None, {"fixture_pos": None}):
        task = ib.make("ib-p1-cracker-box-jaw-v1", obs_mode="proprio", seed=0,
                       config_overrides=overrides)
        try:
            task.reset(seed=0)
            sim = task._env.sim
            fid = task._env._fixture_body_id
            lows = [
                (sim.data.geom_xpos[g] - np.abs(sim.data.geom_xmat[g].reshape(3, 3)) @ sim.model.geom_size[g])[2]
                for g in range(sim.model.ngeom)
                if sim.model.geom_bodyid[g] == fid and sim.model.geom_contype[g]
            ]
            assert min(lows) == pytest.approx(task._table_height, abs=2e-3)
        finally:
            task.close()


def test_slot_box_in_mid_air_not_success():
    """A correctly posed box above the slot centre is not inserted until it has come to rest."""
    task = ib.make("ib-p1-cracker-box-suction-v1", obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        fb = task._env._fixture_bounds
        centre = np.array([0.5 * (lo + hi) for lo, hi in fb.values()])
        _set_free_joint(task, task._env._payload.joints[0], centre + [0.0, 0.0, 0.03],
                        np.array([0.7071068, 0.0, -0.7071068, 0.0]))
        _, _, terminated, _, info = task.step(_zero(task))
        assert terminated is False
        assert info["is_success"] is False
    finally:
        task.close()


def test_slot_reset_frame_shows_tinted_slot():
    """The reset observation already reflects the slot tint (no stale first frame)."""
    task = ib.make("ib-p1-cracker-box-suction-v1", obs_mode="combo", seed=0)
    try:
        obs, _ = task.reset(seed=0)
        fresh = task._rebuild_obs()
        cam_keys = [k for k in obs if k.startswith("rgb_")]
        assert cam_keys
        for k in cam_keys:
            np.testing.assert_array_equal(obs[k], fresh[k])
    finally:
        task.close()


# =====================================================================
#  BI-11: payload start positions are valid
# =====================================================================

def test_cup_spawn_on_table_and_outside_goal():
    task = ib.make("ib-p1-cup-inversion-v1", obs_mode="proprio", seed=0, privileged_obs=True)
    try:
        half = 0.5 * np.asarray(task._env.table_full_size[:2])
        for seed in list(range(0, 100, 7)) + [79]:
            obs, _ = task.reset(seed=seed)
            pos = task._env.sim.data.body_xpos[task._env._payload_body_id]
            np.testing.assert_allclose(obs["object_pos"], pos)
            assert np.all(np.abs(pos[:2]) <= half - task._env._payload.horizontal_radius), seed
            assert not task._payload_in_goal_region(pos), seed
            jitter = np.abs(pos[:2] - np.array([-0.2, 0.0]))
            assert np.all(jitter <= 2.0 * np.array([task._place_x_std, task._place_y_std]) + 1e-6)
    finally:
        task.close()


def test_hobi_plate_spawn_clear_of_shelf():
    task = ib.make("ib-p1-plate-shelf-hobi-v1", obs_mode="proprio", seed=0)
    try:
        for seed in (30, 35, 50, 54, 55, 59, 63, 79, 89, 97):
            task.reset(seed=seed)
            sim = task._env.sim
            plate = sim.model.body_name2id(task._plate.root_body)
            shelf = sim.model.body_name2id(task._shelf.root_body)
            for i in range(sim.data.ncon):
                c = sim.data.contact[i]
                bodies = {int(sim.model.geom_bodyid[c.geom1]), int(sim.model.geom_bodyid[c.geom2])}
                assert not (bodies == {plate, shelf} and c.dist < 0.0), seed
            pos = sim.data.body_xpos[plate]
            assert not task._plate_in_bounds(pos), seed
    finally:
        task.close()


# =====================================================================
#  BI-16: config / doc consistency
# =====================================================================

class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise AssertionError(f"duplicate YAML keys {duplicates} at line {node.start_mark.line + 1}")
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@pytest.mark.parametrize("path", sorted(
    list((CONFIGS_DIR / "hetbi").glob("*.yaml")) + list((CONFIGS_DIR / "hobi").glob("*.yaml"))
), ids=lambda p: p.stem)
def test_bimanual_yaml_has_no_duplicate_keys(path):
    with open(path) as f:
        yaml.load(f, Loader=_UniqueKeyLoader)


# =====================================================================
#  Keyboard: info dict and font fallback
# =====================================================================

def test_keyboard_info_keys():
    task = ib.make("ib-m-keyboard-typing-v1", obs_mode="proprio", seed=0)
    try:
        _, info = task.reset(seed=0)
        assert info["is_success"] is False
        _, _, _, _, info = task.step(_zero(task))
        for key in ("task_score", "is_success", "step", "word", "typed", "next_letter",
                    "applied_action"):
            assert key in info, key
    finally:
        task.close()


def test_keyboard_applied_action_is_the_scripted_arm0_action():
    task = ib.make("ib-m-keyboard-typing-v1", obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        sent = np.full(task.action_space.shape, 0.5, dtype=np.float32)
        for _ in range(5):
            _, _, _, _, info = task.step(sent)
        applied = info["applied_action"]
        assert applied.shape == sent.shape
        np.testing.assert_allclose(applied[7:], sent[7:])
        assert not np.allclose(applied[:7], sent[:7])
    finally:
        task.close()


@pytest.mark.parametrize("module", ["imbench.tasks.hobi.keyboard_typing_v1"])
def test_keyboard_glyphs_render_without_system_font(module, monkeypatch, tmp_path):
    import importlib

    from PIL import Image, ImageFont

    mod = importlib.import_module(module)
    truetype = ImageFont.truetype

    def _no_font_files(font, *args, **kwargs):
        if isinstance(font, (str, bytes)):
            raise OSError("cannot open resource")
        return truetype(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", _no_font_files)
    out = tmp_path / "glyph.png"
    mod._render_glyph_png("A", (0, 0, 0), str(out))
    with Image.open(out) as img:
        assert img.size == (mod._WORD_DISPLAY_GLYPH_PX, mod._WORD_DISPLAY_GLYPH_PX)
        assert np.asarray(img).std() > 0, "glyph image is blank"


# =====================================================================
#  Plate-shelf: success needs a release and the centre slot
# =====================================================================

def test_hobi_plate_held_by_gripper_is_not_success(monkeypatch):
    """A plate standing still in the slot does not succeed while a gripper touches it."""
    task = ib.make("ib-p1-plate-shelf-hobi-v1", obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        monkeypatch.setattr(task, "_plate_released", lambda: False, raising=False)
        fb = task._fixture_bounds
        z = fb["z"][0] + task._plate.horizontal_radius + 0.003
        pos = np.array([np.mean(fb["x"]), np.mean(fb["y"]), z])
        _set_free_joint(task, task._plate.joints[0], pos, _quat_about("x", 90))
        _, _, terminated, _, info = _step_n(task, 80)
        assert task._stage_debug["inserted_in_bounds"] == 1.0
        assert terminated is False
        assert info["is_success"] is False
    finally:
        task.close()


@pytest.mark.parametrize("task_id", ["ib-p1-plate-shelf-hetbi-v1", "ib-p1-plate-shelf-hobi-v1"])
@pytest.mark.parametrize("dy", [0.13, -0.13])
def test_plate_in_side_slot_is_not_success(task_id, dy):
    """The success region is the centre slot: a plate on edge in a side slot earns no insertion."""
    task = ib.make(task_id, obs_mode="proprio", seed=0)
    try:
        task.reset(seed=0)
        hobi = task_id.endswith("hobi-v1")
        fb = (task if hobi else task._env)._fixture_bounds
        joint = task._plate.joints[0] if hobi else task._env._payload.joints[0]
        z = 0.5 * (fb["z"][0] + fb["z"][1])
        pos = np.array([np.mean(fb["x"]), np.mean(fb["y"]) + dy, z])
        _set_free_joint(task, joint, pos, _quat_about("x", 90))
        _, _, _, _, info = _step_n(task, 60)
        assert task._stage_debug["inserted_in_bounds"] == 0.0
        assert info["is_success"] is False
    finally:
        task.close()


# =====================================================================
#  Keyboard: one press types one letter
# =====================================================================

def _keyboard_hello(task_id, monkeypatch, arm0, arm1, typed=0):
    """Reset *task_id* onto HELLO with a fixed split; returns (task, pressed-keys per arm)."""
    task = ib.make(task_id, obs_mode="proprio", seed=0, config_overrides={"word": "HELLO"})
    task.reset(seed=0)
    task._arm0_indices, task._arm1_indices = arm0, arm1
    task._typed_mask = [i < typed for i in range(5)]
    task._typed_count = typed
    pressed = {0: set(), 1: set()}
    monkeypatch.setattr(task, "_keys_pressed_by", lambda arm: set(pressed[arm]))
    return task, pressed


def _press(task, n=None):
    for _ in range(task._press_hold_steps if n is None else n):
        task._detect_key_presses({})


@pytest.mark.parametrize("task_id", ["ib-m-keyboard-typing-v1", "ib-m-keyboard-typing-v2"])
def test_keyboard_one_press_types_one_repeated_letter(task_id, monkeypatch):
    """Both L's of HELLO on arm 1: a held press types one, a fresh press the other."""
    task, pressed = _keyboard_hello(task_id, monkeypatch, [0, 1, 4], [2, 3], typed=2)
    try:
        pressed[1] = {"L"}
        _press(task, 3 * task._press_hold_steps)
        assert task._typed_mask == [True, True, True, False, False]
        assert task._typo_key is None
        pressed[1] = set()
        _press(task, 1)
        pressed[1] = {"L"}
        _press(task)
        assert task._typed_mask == [True, True, True, True, False]
        assert task._typo_key is None
    finally:
        task.close()


@pytest.mark.parametrize("task_id", ["ib-m-keyboard-typing-v1", "ib-m-keyboard-typing-v2"])
def test_keyboard_letters_type_in_word_order(task_id, monkeypatch):
    """Arm 0's later letter cannot type before arm 1's earlier one."""
    task, pressed = _keyboard_hello(task_id, monkeypatch, [0, 2, 4], [1, 3], typed=1)
    try:
        pressed[0] = {"L"}  # arm 0 owns index 2, but index 1 (E, arm 1) is next
        _press(task, 3 * task._press_hold_steps)
        assert task._typed_count == 1
        pressed[0] = set()
        pressed[1] = {"E"}
        _press(task)
        assert task._typed_count == 2 and task._typo_key is None
        # Arm 1 still resting on E after typing it is not a typo.
        _press(task, 3 * task._press_hold_steps)
        assert task._typed_count == 2 and task._typo_key is None
    finally:
        task.close()


@pytest.mark.parametrize("held", ["Q", "L"], ids=["wrong-letter", "own-letter-out-of-turn"])
def test_keyboard_arm1_typo_fails_episode(held, monkeypatch):
    """A held arm-1 press on any key but its next letter ends the episode unsuccessfully."""
    task, pressed = _keyboard_hello("ib-m-keyboard-typing-v1", monkeypatch, [0, 1, 2], [3, 4], typed=1)
    try:
        pressed[1] = {held}  # next letter is E, arm 0's
        _press(task, task._press_hold_steps - 1)
        assert task._typo_key is None and not task._check_termination({})
        _, _, terminated, _, info = task.step(_zero(task))
        assert task._typo_key == held and info["typo"] == held
        assert terminated is True and info["is_success"] is False
        assert task._typed_count == 1
    finally:
        task.close()


def test_keyboard_seeds_move_the_keyboard():
    """Each seed places the keyboard differently, and replays its own placement."""
    task = ib.make("ib-m-keyboard-typing-v1", obs_mode="proprio", seed=0)
    try:
        pos = {}
        for seed in (5, 6, 5):
            task.reset(seed=seed)
            pos.setdefault(seed, []).append(task._keyboard_world_pos.copy())
        np.testing.assert_array_equal(pos[5][0], pos[5][1])
        assert not np.allclose(pos[5][0], pos[6][0])
    finally:
        task.close()


@pytest.mark.slow
def test_keyboard_scripted_demo_types_word_in_order():
    """Scripted arms take turns on an interleaved split and type NEURIPS in order."""
    from data_synthetic.bimanual_pipeline.keyboard_typing_policy import KeyboardTypingPolicy

    task = ib.make("ib-m-keyboard-typing-v1", obs_mode="proprio", seed=0,
                   config_overrides={"word": "NEURIPS",
                                     "arm0_indices": [0, 2, 4, 5, 6], "arm1_indices": [1, 3]})
    try:
        obs, _ = task.reset(seed=6)
        policy = KeyboardTypingPolicy(rng=np.random.default_rng(6))
        typed, done = [], False
        while not done:
            obs, _, terminated, truncated, info = task.step(policy.act(obs, env=task))
            if info["typed"] and (not typed or typed[-1] != info["typed"]):
                typed.append(info["typed"])
            done = terminated or truncated
        assert info["is_success"] and info["typo"] is None
        assert typed == ["NEURIPS"[:i] for i in range(1, 8)]
    finally:
        task.close()


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_keyboard_scripted_demo_succeeds_across_seeds(seed):
    """With the default random word, split and keyboard jitter the scripted
    arms type the drawn word well inside the step limit. (With NEURIPS,
    seeds 1 and 2 needed arm 1 to press bottom-row N at full reach; a
    Cartesian return home then folded its wrist onto the keyboard and
    blocked arm 0.)"""
    from data_synthetic.bimanual_pipeline.keyboard_typing_policy import KeyboardTypingPolicy

    task = ib.make("ib-m-keyboard-typing-v1", obs_mode="proprio", seed=0)
    try:
        obs, _ = task.reset(seed=seed)
        policy = KeyboardTypingPolicy(rng=np.random.default_rng(seed))
        steps, done = 0, False
        while not done:
            obs, _, terminated, truncated, info = task.step(policy.act(obs, env=task))
            steps += 1
            done = terminated or truncated
        assert info["is_success"] and info["typo"] is None
        assert steps < 5000
    finally:
        task.close()


def test_hobi_policy_blocked_insertion_stops_without_releasing(monkeypatch):
    """A blocked insertion fails explicitly instead of releasing outside the shelf."""
    from data_synthetic import mp_hobi_plate_shelf as mp

    task = ib.make(mp.TASK_ID, obs_mode="proprio", privileged_obs=True, seed=0)
    try:
        obs, _ = task.reset(seed=0)
        pol = mp._make_policy(task, obs, np.random.default_rng(0))
        pol._placement, pol._ticks = "insert", 800
        pol._home = obs["robot0_eef_pos"].copy()
        monkeypatch.setattr(task, "_gripper_contacts_plate", lambda arm: True)
        action = pol.act(obs, env=task)
        assert pol.done and pol.failed
        assert pol.failure_reason == "insert target was not reached"
        assert action[6] != -1  # no release command on timeout
    finally:
        task.close()
