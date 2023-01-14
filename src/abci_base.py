import time

import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote

from src.environments.environment import *
from src.experimental_design.exp_designer_base import Design


class ABCIBase:
    def __init__(self, env: Environment, policy='observational', num_workers: int = 1):
        self.env = env
        self.policy = policy

        # init distributed experiment design
        self.num_workers = num_workers
        self.designed_experiments = {}
        self.open_targets = set()
        if num_workers > 1:
            self.worker_id = rpc.get_worker_info().id
            if self.worker_id == 0:
                self.experimenter_rref = RRef(self)
                self.designer_rrefs = []
                for worker_id in range(1, num_workers):
                    info = rpc.get_worker_info(f'ExperimentDesigner{worker_id}')
                    self.designer_rrefs.append(remote(info, self.experiment_designer_factory))

        # init lists
        self.experiments = []
        self.loss_list = []
        self.eshd_list = []
        self.graph_ll_list = []
        self.info_gain_list = []
        self.graph_entropy_list = []
        self.auroc_list = []
        self.auprc_list = []
        self.observational_test_ll_list = []
        self.interventional_test_ll_lists = {node: [] for node in self.env.node_labels}
        self.observational_kld_list = []
        self.interventional_kld_lists = {node: [] for node in self.env.node_labels}
        self.query_kld_list = []

    def experiment_designer_factory(self):
        raise NotImplementedError

    def run(self, num_experiments=10, batch_size=1, update_interval=5, log_interval=5, num_initial_obs_samples=1):
        raise NotImplementedError

    def get_random_intervention(self, fixed_value: float = None):
        target_node = random.choice(list(self.env.intervenable_nodes) + ['OBSERVATIONAL'])
        if target_node == 'OBSERVATIONAL':
            return {}
        if fixed_value is None:
            bounds = self.env.intervention_bounds[target_node]
            target_value = torch.rand(1) * (bounds[1] - bounds[0]) + bounds[0]
        else:
            target_value = torch.tensor(fixed_value)
        return {target_node: target_value}

    def report_design(self, worker_id: int, design_key: str, design: Design):
        print(f'Worker {worker_id} designed {design.interventions} with info gain {design.info_gain}', flush=True)
        self.designed_experiments[design_key] = design

    def report_status(self, worker_id: int, message: str):
        print(f'Worker {worker_id} reports at {time.strftime("%H:%M:%S")}: {message}', flush=True)

    def get_target(self, worker_id: int):
        target = self.open_targets.pop() if self.open_targets else None
        print(f'Worker {worker_id} asks for new target at {time.strftime("%H:%M:%S")}. Assigning new target {target}.',
              flush=True)
        return target

    def design_experiment_distributed(self, args):
        self.open_targets = self.env.intervenable_nodes | {'OBSERVATIONAL'}
        self.designed_experiments.clear()

        # start workers
        futs = []
        for designer_rref in self.designer_rrefs:
            futs.append(designer_rref.rpc_async().run_distributed(self.experimenter_rref, args))

        # init and run designer of master process
        designer = self.experiment_designer_factory()
        designer.init_design_process(args)

        target_node = self.get_target(self.worker_id)
        while target_node:
            design = designer.design_experiment(target_node)
            self.report_design(self.worker_id, target_node, design)
            target_node = self.get_target(self.worker_id)

        # wait until all workers have finished
        for fut in futs:
            fut.wait()

        # pick most promising experiment
        print('Experiment design process has finished:')
        best_intervention = {}
        best_info_gain = self.designed_experiments['OBSERVATIONAL'].info_gain
        for _, design in self.designed_experiments.items():
            print(f'Interventions {design.interventions} expect info gain {design.info_gain}', flush=True)
            if design.info_gain > best_info_gain:
                best_info_gain = design.info_gain
                best_intervention = design.interventions

        return best_intervention, best_info_gain

    def param_dict(self):
        env_param_dict = self.env.param_dict()
        params = {'env_param_dict': env_param_dict,
                  'policy': self.policy,
                  'num_workers': self.num_workers,
                  'experiments': self.experiments,
                  'loss_list': self.loss_list,
                  'eshd_list': self.eshd_list,
                  'graph_ll_list': self.graph_ll_list,
                  'info_gain_list': self.info_gain_list,
                  'graph_entropy_list': self.graph_entropy_list,
                  'auroc_list': self.auroc_list,
                  'auprc_list': self.auprc_list,
                  'observational_test_ll_list': self.observational_test_ll_list,
                  'interventional_test_ll_lists': self.interventional_test_ll_lists,
                  'observational_kld_list': self.observational_kld_list,
                  'interventional_kld_lists': self.interventional_kld_lists,
                  'query_kld_list': self.query_kld_list}
        return params

    def load_param_dict(self, param_dict):
        self.policy = param_dict['policy']
        self.num_workers = param_dict['num_workers']
        self.experiments = param_dict['experiments']
        self.loss_list = param_dict['loss_list']
        self.eshd_list = param_dict['eshd_list']
        self.graph_ll_list = param_dict['graph_ll_list']
        self.info_gain_list = param_dict['info_gain_list']
        self.graph_entropy_list = param_dict['graph_entropy_list']
        self.auroc_list = param_dict['auroc_list']
        self.auprc_list = param_dict['auprc_list']
        self.observational_test_ll_list = param_dict['observational_test_ll_list']
        self.interventional_test_ll_lists = param_dict['interventional_test_ll_lists']
        self.observational_kld_list = param_dict['observational_kld_list']
        self.interventional_kld_lists = param_dict['interventional_kld_lists']
        self.query_kld_list = param_dict['query_kld_list']
