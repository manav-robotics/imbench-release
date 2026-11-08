"""IMBench environments.

Each environment lives in its own module and is auto-registered with
robosuite via its ``EnvMeta`` metaclass.  Import this package to ensure
all environments are available.

Usage::

    from imbench.envs import HetBiEnv          # direct class
    import robosuite; robosuite.make("HetBiEnv", …)  # also works
"""

from imbench.envs.hetbi_env import HetBiEnv
from imbench.envs.hetbi_cup_inversion_env import HetBiCupInversionEnv
from imbench.envs.hobi_env import HoBiEnv
from imbench.envs.edge_slide_env import EdgeSlideEnv
from imbench.envs.single_arm_table_env import SingleArmTableEnv

__all__ = [
    "HetBiEnv",
    "HetBiCupInversionEnv",
    "HoBiEnv",
    "EdgeSlideEnv",
    "SingleArmTableEnv",
]
