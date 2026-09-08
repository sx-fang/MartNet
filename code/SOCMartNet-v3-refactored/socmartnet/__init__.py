"""SOC-MartNet v3 refactored baseline package (arXiv:2405.03169v3)."""

from .networks import DNNtx, control_net, value_net, test_net
from .solver import SOCMartNet, bat_vgrad
from .problems import (HJBLQ, LinearParabolicSin, LinearSinAC, NonDegHJB,
                       ShiftTargetHJB, diag_points, e1_points,
                       manifold_points, origin_point, region_grid,
                       relative_l1, segment_grid)
from . import evaluate

__all__ = [
    'DNNtx', 'control_net', 'value_net', 'test_net',
    'SOCMartNet', 'bat_vgrad',
    'LinearParabolicSin', 'NonDegHJB', 'segment_grid', 'relative_l1',
    'HJBLQ', 'ShiftTargetHJB', 'LinearSinAC',
    'e1_points', 'diag_points', 'manifold_points', 'origin_point',
    'region_grid', 'evaluate',
]
