import time
from collections import namedtuple
from typing import Dict, Tuple, Set

import torch.distributed.rpc as rpc
import torch.optim

from src.experimental_design.optimization import random_search, grid_search, gp_ucb

Design = namedtuple('Design', ['interventions', 'info_gain'])


class ExpDesignerBase:
    def __init__(self, intervention_bounds: Dict[str, Tuple[float, float]], opt_strategy: str = 'gp-ucb',
                 distributed=False):
        self.worker_id = rpc.get_worker_info().id if distributed else 0
        self.intervention_bounds = intervention_bounds
        if opt_strategy not in {'gp-ucb', 'random', 'grid'}:
            print('Invalid optimization strategy ' + opt_strategy + '. Doing Bayesian optimization instead.')
            opt_strategy = 'gp-ucb'
        self.opt_strategy = opt_strategy
        self.utility = None

    def init_design_process(self, args: dict):
        raise NotImplementedError

    def run_distributed(self, experimenter_rref, args: dict):
        experimenter_rref.rpc_sync().report_status(self.worker_id, 'Initializing design process...')
        self.init_design_process(args)
        experimenter_rref.rpc_sync().report_status(self.worker_id, 'Finished initializing design process...')

        target_node = experimenter_rref.rpc_sync().get_target(self.worker_id)
        while target_node:
            design = self.design_experiment(target_node)
            experimenter_rref.rpc_sync().report_design(self.worker_id, target_node, design)
            target_node = experimenter_rref.rpc_sync().get_target(self.worker_id)

    def design_experiment(self, target_node: str):

        # if no target is given report info gain of observational sample
        if target_node == 'OBSERVATIONAL':
            try:
                score = self.utility({})
            except Exception as e:
                print(f'Exception occured in ExperimentDesigner.design_experiment() when the score for the '
                      f'observational target:')
                print(e)
                score = torch.tensor(0.)
            return Design({}, score)

        # otherwise, design experiment for target node
        bounds = torch.Tensor(self.intervention_bounds[target_node]).view(2, 1)
        if self.opt_strategy == 'random':
            target_value, score = random_search(lambda x: self.utility({target_node: x}), bounds)
        elif self.opt_strategy == 'grid':
            target_value, score = grid_search(lambda x: self.utility({target_node: x}), bounds)
        else:
            target_value, score = gp_ucb(lambda x: self.utility({target_node: x}), bounds)

        return Design({target_node: target_value}, score)

    def get_best_experiment(self, target_nodes: Set[str]):
        best_intervention = {}
        best_score = self.utility({})
        print(f'Expected information gain for observational sample is {best_score}.')
        for target_node in target_nodes:
            print(f'Start experiment design for node {target_node} at {time.strftime("%H:%M:%S")}')
            design = self.design_experiment(target_node)
            print(f'Expected information gain for {design.interventions} is {design.info_gain}.')
            if design.info_gain > best_score:
                best_score = design.info_gain
                best_intervention = design.interventions

        return best_intervention, best_score
