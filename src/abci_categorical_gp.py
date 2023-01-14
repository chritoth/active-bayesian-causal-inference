from typing import Callable

import torch.optim

from src.abci_base import ABCIBase
from src.environments.environment import *
from src.experimental_design.exp_designer_abci_categorical_gp import ExpDesignerABCICategoricalGP
from src.models.gp_model import GaussianProcessModel
from src.models.graph_models import CategoricalModel, graph_to_adj_mat
from src.utils.metrics import shd, auroc, auprc


class ABCICategoricalGP(ABCIBase):
    policies = {'observational', 'random', 'random-fixed-value', 'graph-info-gain', 'scm-info-gain',
                'intervention-info-gain', 'oracle'}

    def __init__(self, env: Environment, policy, num_workers: int = 1, linear: bool = False):
        assert policy in self.policies, print(f'Invalid policy {policy}!')
        super().__init__(env, policy, num_workers)

        # store params
        self.linear = linear

        # init models
        self.graph_prior = CategoricalModel(self.env.node_labels)
        self.graph_posterior = CategoricalModel(self.env.node_labels)
        self.mechanism_model = GaussianProcessModel(env.node_labels, linear=linear)

        # init mechanisms for all graphs
        for graph in self.graph_prior.graphs:
            self.mechanism_model.init_mechanisms(graph)

    def experiment_designer_factory(self):
        distributed = self.num_workers > 1
        return ExpDesignerABCICategoricalGP(self.env.intervention_bounds, opt_strategy='gp-ucb',
                                            distributed=distributed)

    def run(self, num_experiments=10, batch_size=3, update_interval=5, log_interval=1, num_initial_obs_samples=1,
            checkpoint_interval: int = 10, outdir: str = None, job_id: str = ''):

        # pre-compute env test data stats
        with torch.no_grad():
            num_test_samples = self.env.num_test_samples_per_intervention
            env_obs_test_ll = self.env.log_likelihood(self.env.observational_test_data) / num_test_samples
            env_intr_test_lls = {}
            for node, experiments in self.env.interventional_test_data.items():
                env_intr_test_lls[node] = self.env.log_likelihood(experiments) / num_test_samples

        # run experiments
        for epoch in range(num_experiments):
            print(f'Starting experiment cycle {epoch + 1}/{num_experiments}...')

            # pick intervention according to policy
            print(f'Design and perform experiment...', flush=True)
            info_gain = None
            if self.policy == 'observational' or len(self.experiments) == 0 and num_initial_obs_samples > 0:
                interventions = {}
            elif self.policy == 'random':
                interventions = self.get_random_intervention()
            elif self.policy == 'random-fixed-value':
                interventions = self.get_random_intervention(0.)
            elif self.policy == 'oracle':
                interventions, info_gain = self.get_oracle_intervention(batch_size)
            else:
                if self.policy == 'graph-info-gain' or self.policy == 'scm-info-gain':
                    args = {'mechanism_model': self.mechanism_model,
                            'graph_posterior': self.graph_posterior,
                            'batch_size': batch_size,
                            'num_exp_per_graph': 500,
                            'policy': self.policy,
                            'mode': 'full'}
                elif self.policy == 'intervention-info-gain':
                    outer_mc_graphs, outer_log_weights = self.graph_posterior.get_mc_graphs('n-best', 5)
                    args = {'mechanism_model': self.mechanism_model,
                            'graph_posterior': self.graph_posterior,
                            'experiments': self.experiments,
                            'interventional_queries': self.env.interventional_queries,
                            'outer_mc_graphs': outer_mc_graphs,
                            'outer_log_weights': outer_log_weights,
                            'num_mc_queries': 5,
                            'num_batches_per_query': 3,
                            'batch_size': batch_size,
                            'num_exp_per_graph': 50,
                            'policy': self.policy,
                            'mode': 'full'}
                else:
                    assert False, print(f'Invalid policy {self.policy}!')

                if self.num_workers > 1:
                    interventions, info_gain = self.design_experiment_distributed(args)
                else:
                    designer = self.experiment_designer_factory()
                    designer.init_design_process(args)
                    interventions, info_gain = designer.get_best_experiment(self.env.intervenable_nodes)

            # record expected information gain of chosen intervention
            if info_gain is None:
                info_gain = torch.tensor(-1.)
            self.info_gain_list.append(info_gain)

            # perform experiment
            num_experiments_conducted = len(self.experiments)
            num_samples = batch_size
            if num_experiments_conducted == 0 and num_initial_obs_samples > 0:
                num_samples = num_initial_obs_samples
            self.experiments.append(self.env.sample(interventions, num_samples))

            # set training data for mechanisms
            self.mechanism_model.set_data(self.experiments)

            # update mechanism hyperparameters
            hyperparam_update_interval = 1
            if hyperparam_update_interval > 0:
                if num_experiments_conducted <= 1:
                    update_time = 1
                else:
                    update_time = (num_experiments_conducted // hyperparam_update_interval) * hyperparam_update_interval

                self.mechanism_model.update_gp_hyperparameters(update_time, self.experiments, set_data=True)

            # update graph posterior
            if not self.policy == 'oracle':
                print(f'Updating graph posterior...', flush=True)
                self.graph_posterior = self.compute_graph_posterior(self.experiments, use_cache=True)

            print(f'Logging evaluation stats...', flush=True)

            # record graph posterior entropy
            self.graph_entropy_list.append(self.graph_posterior.entropy().detach())

            # record expected SHD
            with torch.no_grad():
                eshd = self.graph_posterior_expectation(lambda g: shd(self.env.graph, g))
            self.eshd_list.append(eshd)

            # record env graph LL
            self.graph_ll_list.append(self.graph_posterior.log_prob(self.env.graph))

            # record observational test LLs
            def test_ll(graph):
                return self.mechanism_model.mll(self.env.observational_test_data, graph, prior_mode=False,
                                                use_cache=True, mode='joint')

            self.mechanism_model.clear_posterior_mll_cache()
            with torch.no_grad():
                ll = self.graph_posterior_expectation(test_ll)
            self.observational_test_ll_list.append(ll)

            # record interventional test LLs
            for node, experiments in self.env.interventional_test_data.items():
                def test_ll(graph):
                    return self.mechanism_model.mll(experiments, graph, prior_mode=False, use_cache=True, mode='joint')

                self.mechanism_model.clear_posterior_mll_cache()
                with torch.no_grad():
                    ll = self.graph_posterior_expectation(test_ll)
                self.interventional_test_ll_lists[node].append(ll)

            # record observational KLD
            def test_ll(graph):
                return self.mechanism_model.mll(self.env.observational_test_data, graph, prior_mode=False,
                                                use_cache=True, mode='independent_samples', reduce=False)

            self.mechanism_model.clear_posterior_mll_cache()
            with torch.no_grad():
                ll = self.graph_posterior_expectation(test_ll, logspace=True).mean()
            self.observational_kld_list.append(env_obs_test_ll - ll)

            # record interventional test KLDs
            for node, experiments in self.env.interventional_test_data.items():
                def test_ll(graph):
                    return self.mechanism_model.mll(experiments, graph, prior_mode=False, use_cache=True,
                                                    mode='independent_samples', reduce=False)

                self.mechanism_model.clear_posterior_mll_cache()
                with torch.no_grad():
                    ll = self.graph_posterior_expectation(test_ll, logspace=True).mean()
                self.interventional_kld_lists[node].append(env_intr_test_lls[node] - ll)

            # record AUROC/AUPRC scores
            with torch.no_grad():
                posterior_edge_probs = self.graph_posterior.edge_probs()
                true_adj_mat = graph_to_adj_mat(self.env.graph, self.env.node_labels)
            self.auroc_list.append(auroc(posterior_edge_probs, true_adj_mat))
            self.auprc_list.append(auprc(posterior_edge_probs, true_adj_mat))

            # record query KLD
            if self.env.interventional_queries is not None:
                def test_query_ll(graph):
                    query_lls = self.mechanism_model.query_log_probs(self.env.interventional_queries, graph, 200)
                    return query_lls

                self.mechanism_model.clear_posterior_mll_cache()
                with torch.no_grad():
                    ll = self.graph_posterior_expectation(test_query_ll, logspace=True).mean()
                self.query_kld_list.append(self.env.query_ll - ll)

            if outdir is not None and 0 < epoch < num_experiments - 1 and (epoch + 1) % checkpoint_interval == 0:
                model = 'abci-categorical-gp-linear' if self.linear else 'abci-categorical-gp'
                outpath = outdir + model + '-' + self.policy + '-' + self.env.name + f'-{job_id}-exp-{epoch + 1}.pth'
                self.save(outpath)

            if log_interval > 0 and epoch % log_interval == 0:
                print(f'Experiment {epoch + 1}/{num_experiments}, ESHD is {eshd.item()}', flush=True)

    def get_oracle_intervention(self, num_samples: int, num_candidates_per_node: int = 10):
        current_entropy = self.graph_posterior.entropy()

        self.experiments.append(self.env.sample({}, num_samples))
        posterior = self.compute_graph_posterior(self.experiments)
        best_intervention = {}
        best_info_gain = current_entropy - posterior.entropy()
        best_posterior = posterior

        for node in self.env.node_labels:
            bounds = self.env.intervention_bounds[node]
            candidates = torch.linspace(bounds[0], bounds[1], num_candidates_per_node)
            for i in range(num_candidates_per_node):
                self.experiments[-1] = self.env.sample({node: candidates[i]}, num_samples)
                posterior = self.compute_graph_posterior(self.experiments)
                info_gain = current_entropy - posterior.entropy()
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_intervention = {node: candidates[i]}
                    best_posterior = posterior

        self.graph_posterior = best_posterior
        return best_intervention, best_info_gain

    def compute_graph_posterior(self, experiments: List[Experiment], use_cache: bool = False) -> CategoricalModel:
        posterior = CategoricalModel(self.env.node_labels)
        self.mechanism_model.clear_prior_mll_cache()
        with torch.no_grad():
            for graph in posterior.graphs:
                mll = self.mechanism_model.mll(experiments, graph, prior_mode=True, use_cache=use_cache).squeeze()
                posterior.set_log_prob(mll + self.graph_prior.log_prob(graph), graph)

            posterior.normalize()
        return posterior

    def graph_posterior_expectation(self, func: Callable[[nx.DiGraph], torch.Tensor], logspace=False):
        with torch.no_grad():
            # compute function values
            func_values = [func(graph) for graph in self.graph_posterior.graphs]
            func_output_shape = func_values[0].shape
            func_output_dim = len(func_output_shape)
            func_values = torch.stack(func_values).view(self.graph_posterior.num_graphs, *func_output_shape)

            # compute expectation
            if logspace:
                log_graph_weights = torch.tensor([self.graph_posterior.log_prob(graph) for graph in
                                                  self.graph_posterior.graphs])
                log_graph_weights = log_graph_weights.view(self.graph_posterior.num_graphs, *([1] * func_output_dim))

                expected_value = (log_graph_weights + func_values).logsumexp(dim=0)
                return expected_value

            graph_weights = torch.tensor([self.graph_posterior.log_prob(graph).exp() for graph in
                                          self.graph_posterior.graphs])
            graph_weights = graph_weights.view(self.graph_posterior.num_graphs, *([1] * func_output_dim))

            expected_value = (graph_weights * func_values).sum(dim=0)
            return expected_value

    def param_dict(self):
        params = super().param_dict()
        params.update({'linear': self.linear,
                       'mechanism_model_params': self.mechanism_model.param_dict(),
                       'graph_prior_params': self.graph_prior.param_dict(),
                       'graph_posterior_params': self.graph_posterior.param_dict()})
        return params

    def load_param_dict(self, param_dict):
        super().load_param_dict(param_dict)
        self.linear = param_dict['linear']
        self.mechanism_model.load_param_dict(param_dict['mechanism_model_params'])
        self.graph_prior.load_param_dict(param_dict['graph_prior_params'])
        self.graph_posterior.load_param_dict(param_dict['graph_posterior_params'])

    def save(self, path):
        torch.save(self.param_dict(), path)

    @classmethod
    def load(cls, path, num_workers: int = 1):
        param_dict = torch.load(path)

        env_param_dict = param_dict['env_param_dict']
        env = Environment(env_param_dict['num_nodes'], mechanism_model=None, num_test_samples_per_intervention=0,
                          num_test_queries=0, graph=env_param_dict['graph'])
        env.load_param_dict(env_param_dict)

        abci = ABCICategoricalGP(env, param_dict['policy'], num_workers, param_dict['linear'])
        abci.load_param_dict(param_dict)
        return abci
