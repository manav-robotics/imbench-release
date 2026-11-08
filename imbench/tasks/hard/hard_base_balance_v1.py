"""Hard-Base Balance — balance an elongated rod on a thinner, shorter ridge.

Same physics and scoring as :class:`MediumBaseBalanceV1`; only the YAML
config differs (narrower and shorter ridge → less forgiving balancing).

Task ID: ``ib-p7-balance-hard-v1``
"""

from imbench.tasks.hard.medium_base_balance_v1 import MediumBaseBalanceV1


class HardBaseBalanceV1(MediumBaseBalanceV1):
    TASK_ID = "ib-p7-balance-hard-v1"
    DESCRIPTION = (
        "Single arm: balance an elongated rod (pen/rod) vertically on a "
        "narrow perpendicular base. Harder variant with a thinner and "
        "shorter ridge — less forgiving than the medium variant."
    )
