"""Contact queries shared by task scorers."""

import re

import numpy as np

#: Body names of everything a robot is made of: arm links, hand, fingers and
#: tactile pads (``robot<i>_*`` / ``gripper<i>_*`` for every robot index).
ROBOT_BODY_PATTERN = re.compile(r"(robot|gripper)\d+_")


def robot_geom_mask(sim):
    """Boolean mask over geom ids, True for geoms on any robot's arm, hand, fingers or tactile pads."""
    model = sim.model
    robot_bodies = [
        b for b in range(model.nbody)
        if ROBOT_BODY_PATTERN.match(model.body_id2name(b) or "")
    ]
    return np.isin(model.geom_bodyid, robot_bodies)


def _subtree_mask(model, root_body):
    root = model.body_name2id(root_body)
    in_subtree = np.zeros(model.nbody, dtype=bool)
    in_subtree[root] = True
    # MuJoCo orders bodies so that a parent always precedes its children.
    for b in range(root + 1, model.nbody):
        in_subtree[b] = in_subtree[model.body_parentid[b]]
    return in_subtree


def body_subtree_ids(sim, root_body):
    """Ids of *root_body* (a body name) and all of its descendant bodies."""
    return set(np.flatnonzero(_subtree_mask(sim.model, root_body)).tolist())


def body_geom_ids(sim, root_body):
    """Ids of every geom on *root_body* (a body name) and its descendants."""
    model = sim.model
    return set(np.flatnonzero(_subtree_mask(model, root_body)[model.geom_bodyid]).tolist())


def object_geom_ids(env, name):
    """Ids of every geom of the env object called *name*."""
    return body_geom_ids(env.sim, env.get_object(name).root_body)


def robot_contacts(sim, geom_ids):
    """Whether any robot geom currently touches a geom in *geom_ids*."""
    ncon = sim.data.ncon
    if not geom_ids or ncon == 0:
        return False
    robot = robot_geom_mask(sim)
    for contact in sim.data.contact[:ncon]:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if (robot[g1] and g2 in geom_ids) or (robot[g2] and g1 in geom_ids):
            return True
    return False


def geoms_touch(sim, geoms_a, geoms_b):
    """Whether any geom in *geoms_a* currently touches a geom in *geoms_b*."""
    for contact in sim.data.contact[:sim.data.ncon]:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b):
            return True
    return False
