"""Cascade-order scoring shared by the domino tasks."""

import numpy as np

from imbench.tasks.prehensile.contacts import geoms_touch


class DominoCascadeMixin:
    """Credits a target displaced only by an unbroken, in-order domino cascade.

    The host task sets ``_cascade_geoms`` (one geom-id set per body, from the
    first domino to the target), ``_link_touch_step`` (one entry per
    consecutive pair, ``-1`` until the pair first touches) and
    ``_first_domino_toppled``, and defines ``_target_displaced()``.
    """

    def _update_cascade_links(self):
        """Record the step at which each consecutive pair in cascade order first touches."""
        sim = self._env.sim
        for k in np.flatnonzero(self._link_touch_step < 0):
            if geoms_touch(sim, self._cascade_geoms[k], self._cascade_geoms[k + 1]):
                self._link_touch_step[k] = self._step_count

    def _cascade_displaced_target(self):
        """Whether the target is displaced at the end of an unbroken, in-order cascade.

        The first domino must have toppled and every consecutive pair from it
        to the target must have touched, each no earlier than the pair before
        it (pairs already touching at reset are exempt from the order). A
        domino thrown at the target or the chain's end, or a chain knocked
        over backwards from the target, does not count.
        """
        steps = self._link_touch_step
        linked = bool((steps >= 0).all() and (np.diff(steps[steps > 0]) >= 0).all())
        return self._first_domino_toppled and linked and self._target_displaced()
