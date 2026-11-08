"""One physical Y-merge: an indirect feeder joins a broken straight lane."""
import numpy as np


def merge_layout(task):
    rng = task._rng
    a = task._correct_chain
    b, c, d, e = rng.permutation([k for k in range(task._num_chains) if k != a])[:4]
    task._receiver_chain, task._false_feeder_chain = int(b), int(d)
    n = task._dominoes_per_chain
    def name(k, j):
        return f"chain{k}_domino_{j}"
    variations = task._episode_variations
    headings = np.deg2rad([0, 0, 15, 30, 45, 45, 45, 45])
    if variations["shape"]:
        headings[1:4] = np.deg2rad([[0, 15, 30], [10, 25, 40], [15, 30, 45]][rng.randint(3)])
    spacing = task._correct_spacing
    points = np.vstack([np.zeros(2), np.cumsum(spacing*np.c_[np.cos(headings), np.sin(headings)], axis=0)])
    points[:, 0] -= .22
    points[:, 1] -= points[-1, 1]
    route = [name(a, j) for j in range(6)] + [name(c, 4), name(c, 5)]
    false_route = [name(d, j) for j in range(6)] + [name(c, 2), name(c, 3)]
    layout = {}
    for j, (good, bad) in enumerate(zip(route, false_route)):
        h = headings[j]
        layout[good] = (*points[j], h)
        layout[bad] = (points[j, 0], -points[j, 1], -h)
    merge = int(rng.choice([2, 3]))
    task._merge_index = merge
    for j in range(6):
        if j < merge:
            layout[name(b, j)] = (-.30+j*spacing, 0, 0)
        else:
            layout[name(b, j)] = (points[-1, 0]+(j-merge)*spacing, 0, 0)
            route.append(name(b, j))
    for j in range(2):
        layout[name(c, j)] = (-.30+j*spacing, -.32, 0)
    for j in range(6):
        layout[name(e, j)] = (-.30+j*spacing+(.14 if j >= 2 else 0), .32, 0)
    # Both feeders look connected to the same merge. Break only the false
    # feeder upstream; low pairs retain its full visible path/length.
    defect = str(rng.choice(task._network_defects)) if variations["interruption"] else "gap"
    split = int(rng.choice([2, 3]))
    if defect == "short":
        task._chain_sizes[d, split:split+2, 0] = task._domino_size[0]*.23
    elif defect == "missing":
        for j in (2, 3, 4):
            layout.pop(name(d, j))
    else:
        for j in range(2):
            x, y, h = layout[name(d, j)]
            layout[name(d, j)] = (x-.14, y, h)
    # A second broken straight distractor stays remote from the merge.
    mirror = int(rng.choice([-1, 1]))
    yaw = task._LAYOUT_YAW
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    shift = rng.uniform(-.012, .012, 2)
    for key, (x, y, h) in layout.items():
        xy = np.array([x, mirror*y]) @ rotation.T + shift
        layout[key] = (float(xy[0]), float(xy[1]), float(mirror*h+yaw))
    task._solution_route = route
    task._network_defect = defect
    task._network_mirror = mirror
    task._network_layout = layout
    task._parked = {name(k, j) for k in range(task._num_chains) for j in range(n)} - set(layout)
    task._resize_dominoes()
    from imbench.tasks.prehensile.domino_patterns import heading_to_quat
    for key, (x, y, h) in layout.items():
        k, j = (int(v) for v in key.replace("chain", "").split("_domino_"))
        task._place_object(key, [x, y, task._table_height+task._chain_sizes[k, j, 0]+.002], quat=heading_to_quat(h))
    task._park_objects(task._parked)
    task._chain_specs = [{"pattern": "merge", "defect": "none" if k == a else defect,
                          "layout": [layout.get(name(k, j), (100., 100., 0.)) for j in range(n)]}
                         for k in range(task._num_chains)]
    return [layout[key] for key in route]
