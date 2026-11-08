"""Physical placement precision with the actual two-second success hold."""
import numpy as np

import imbench as ib


def _trial(offset, *, baseline=False):
    override = {"task": {"seesaw_damping": .03, "seesaw_frictionloss": .05}} if baseline else None
    task = ib.make("ib-p3-seesaw-balance-v1", obs_mode="proprio", config_overrides=override)
    try:
        task.reset(seed=0)
        # A controlled placement probe: only initialize the counterweight,
        # then let unmodified contacts/gravity/hinge dynamics determine balance.
        task._place_object("weight_0", [task._fulcrum_x-task._ideal_counter_arm-offset,
                           0, task._table_height+task._fulcrum_height+task._seesaw_thickness+.025])
        task._set_object_velocity("weight_0", [0, 0, 0], [0, 0, 0])
        task._env.sim.forward()
        hold = 0
        for _ in range(600):
            _, _, term, trunc, info = task.step(np.array([0, 0, 0, 0, 0, 0, -1.]))
            hold = max(hold, task._balance_counter)
            if term or trunc:
                break
        return bool(info["is_success"]), task._get_seesaw_angle_deg(), hold
    finally:
        task.close()


def test_centimetre_error_previously_hidden_by_hinge_friction_now_tips():
    before = _trial(.02, baseline=True)
    after = _trial(.02)
    assert before[0] and before[1] < 5
    assert not after[0] and after[1] > 5
    assert after[2] < 200


def test_correct_lever_arm_still_completes_full_default_hold():
    success, angle, hold = _trial(0)
    assert success and angle < 5 and hold == 200
