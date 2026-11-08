"""Fully-scripted bimanual motion policies for all ib-p1-*-hetbi-* tasks.

Each module exposes a single ``PolicyInterface`` subclass that reads live
object/fixture positions from the simulation and uses proportional Cartesian
control to execute the task without any planning or learning.

Usage::

    python -m data_synthetic.hetbi_scripted.collect \\
        --task ib-p1-plate-shelf-hetbi-v1 --n-episodes 5

    # or run all hetbi tasks:
    python -m data_synthetic.hetbi_scripted.collect --all --n-episodes 5
"""
from data_synthetic.hetbi_scripted.plate_shelf_policy import PlateShelfPolicy
from data_synthetic.hetbi_scripted.bowl_table_policy import BowlTablePolicy
from data_synthetic.hetbi_scripted.cup_inversion_policy import CupInversionPolicy
from data_synthetic.hetbi_scripted.cracker_box_suction_policy import CrackerBoxSuctionPolicy
from data_synthetic.hetbi_scripted.cracker_box_jaw_policy import CrackerBoxJawPolicy

TASK_POLICY_MAP = {
    "ib-p1-plate-shelf-hetbi-v1":       PlateShelfPolicy,
    "ib-p1-bowl-table-v1":        BowlTablePolicy,
    "ib-p1-cup-inversion-v1":     CupInversionPolicy,
    "ib-p1-cracker-box-suction-v1": CrackerBoxSuctionPolicy,
    "ib-p1-cracker-box-jaw-v1": CrackerBoxJawPolicy,
}

__all__ = [
    "PlateShelfPolicy",
    "BowlTablePolicy",
    "CupInversionPolicy",
    "CrackerBoxSuctionPolicy",
    "CrackerBoxJawPolicy",
    "TASK_POLICY_MAP",
]
