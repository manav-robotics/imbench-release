"""Hard-Base Balance (random angle) — balance a rod on a thinner, shorter ridge.

Same physics, scoring and per-episode random in-plane orientation as
:class:`MediumBaseBalanceV2`; only the YAML config differs (narrower and
shorter ridge → less forgiving balancing).

Task ID: ``ib-p7-balance-hard-v2``
"""

from imbench.tasks.hard.medium_base_balance_v2 import MediumBaseBalanceV2


class HardBaseBalanceV2(MediumBaseBalanceV2):
    TASK_ID = "ib-p7-balance-hard-v2"
    DESCRIPTION = (
        "OOD variant of ib-p7-balance-hard-v1: balance an elongated rod "
        "vertically on a thin, short ridge whose in-plane orientation is "
        "sampled afresh each episode."
    )
