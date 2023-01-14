from typing import Dict, Tuple, List, Optional

import networkx as nx
import torch.optim

from src.environments.environment import Experiment, InterventionalDistributionsQuery
from src.experimental_design.exp_designer_base import ExpDesignerBase
from src.models.gp_model import GaussianProcessModel
from src.models.graph_models import CategoricalModel


class ExpDesignerABCICategoricalGP(ExpDesignerBase):
    model: Optional[GaussianProcessModel]
    graph_posterior: Optional[CategoricalModel]

    def __init__(self, intervention_bounds: Dict[str, Tuple[float, float]], opt_strategy: str = 'gp-ucb',
                 distributed=False):
        super().__init__(intervention_bounds, opt_strategy, distributed)
        self.model = None
        self.graph_posterior = None

    def init_design_process(self, args: dict):
        self.model = args['mechanism_model']
        self.model.entropy_cache.clear()
        self.graph_posterior = args['graph_posterior']

        if args['policy'] == 'scm-info-gain':
            def utility(interventions: dict):
                return self.scm_info_gain(interventions, args['batch_size'], args['num_exp_per_graph'], args['mode'])
        elif args['policy'] == 'graph-info-gain':
            def utility(interventions: dict):
                return self.graph_info_gain(interventions, args['batch_size'], args['num_exp_per_graph'], args['mode'])
        elif args['policy'] == 'intervention-info-gain':
            def utility(interventions: dict):
                return self.intervention_info_gain(interventions,
                                                   args['experiments'],
                                                   args['interventional_queries'],
                                                   args['outer_mc_graphs'],
                                                   args['outer_log_weights'],
                                                   args['num_mc_queries'],
                                                   args['num_batches_per_query'],
                                                   args['batch_size'],
                                                   args['num_exp_per_graph'],
                                                   args['mode'])
        else:
            assert False, 'Invalid policy ' + args['policy'] + '!'
        self.utility = utility

    def compute_graph_posterior_mlls(self, experiments: List[Experiment], graphs: List[nx.DiGraph],
                                     mode='independent_batches', reduce=True):
        graph_mlls = [self.model.mll(experiments, graph, prior_mode=False, use_cache=True, mode=mode, reduce=reduce) for
                      graph in graphs]
        graph_mlls = torch.stack(graph_mlls)
        return graph_mlls

    def graph_info_gain(self, interventions: dict, batch_size: int = 1, num_exp_per_graph: int = 1, mode='n-best',
                        num_mc_graphs=20):
        with torch.no_grad():
            graphs, log_weights = self.graph_posterior.get_mc_graphs(mode, num_mc_graphs)

            graph_info_gains = torch.zeros(len(graphs))
            for graph_idx, graph in enumerate(graphs):
                simulated_experiments = [self.model.sample(interventions, batch_size, num_exp_per_graph, graph)]

                self.model.clear_posterior_mll_cache()
                posterior_mlls = self.compute_graph_posterior_mlls(simulated_experiments, graphs, reduce=False)
                data_mll = (posterior_mlls + log_weights.unsqueeze(-1)).logsumexp(dim=0)
                graph_info_gains[graph_idx] += (posterior_mlls[graph_idx] - data_mll).mean()

        return log_weights.exp() @ graph_info_gains

    def scm_info_gain(self, interventions: dict, batch_size: int = 1, num_exp_per_graph: int = 1, mode='n-best',
                      num_mc_graphs=20):
        with torch.no_grad():
            graphs, log_weights = self.graph_posterior.get_mc_graphs(mode, num_mc_graphs)

            graph_info_gains = torch.zeros(len(graphs))
            for graph_idx, graph in enumerate(graphs):
                simulated_experiments = [self.model.sample(interventions, batch_size, num_exp_per_graph, graph)]

                self.model.clear_posterior_mll_cache()
                posterior_mlls = self.compute_graph_posterior_mlls(simulated_experiments, graphs, reduce=False)
                data_mll = (posterior_mlls + log_weights.unsqueeze(-1)).logsumexp(dim=0)
                entropy = self.model.expected_noise_entropy(interventions, graph, use_cache=True)
                graph_info_gains[graph_idx] += -entropy * batch_size - data_mll.mean()

        return log_weights.exp() @ graph_info_gains

    def intervention_info_gain(self,
                               interventions: dict,
                               experiments: List[Experiment],
                               interventional_queries: List[InterventionalDistributionsQuery],
                               outer_mc_graphs: List[nx.DiGraph],
                               outer_log_weights: torch.Tensor,
                               num_mc_queries: int = 1,
                               num_batches_per_query: int = 1,
                               batch_size: int = 1,
                               num_exp_per_graph: int = 1,
                               mode='n-best', num_inner_mc_graphs=20):
        with torch.no_grad():
            # get mc graphs and compute log weights
            inner_mc_graphs, inner_log_weights = self.graph_posterior.get_mc_graphs(mode, num_inner_mc_graphs)
            num_outer_graphs = len(outer_mc_graphs)
            num_inner_graphs = len(inner_mc_graphs)

            graph_info_gains = torch.zeros(num_outer_graphs)
            for graph_idx, graph in enumerate(outer_mc_graphs):
                # simulate experiments and compute data mll
                self.model.set_data(experiments)
                self.model.clear_posterior_mll_cache()
                simulated_experiments = [self.model.sample(interventions, batch_size, num_exp_per_graph, graph)]
                posterior_mlls = self.compute_graph_posterior_mlls(simulated_experiments, inner_mc_graphs, reduce=False)
                assert posterior_mlls.shape == (num_inner_graphs, num_exp_per_graph)
                data_mll = (posterior_mlls + inner_log_weights.unsqueeze(-1)).logsumexp(dim=0).mean()

                # simulate queries
                self.model.set_data(experiments + simulated_experiments)
                self.model.clear_posterior_mll_cache()

                # simulate queries & compute query lls
                mc_queries = self.model.sample_queries(interventional_queries, num_mc_queries,
                                                       num_batches_per_query, graph)

                query_lls = torch.stack([self.model.query_log_probs(mc_queries, g) for g in inner_mc_graphs])
                num_mc_queries = len(mc_queries[0].sample_queries)
                num_batches_per_query = mc_queries[0].sample_queries[0].num_batches
                assert query_lls.shape == (num_inner_graphs, num_mc_queries, num_batches_per_query)

                # compute and update per graph info gain
                tmp = posterior_mlls.view(num_inner_graphs, 1, 1, num_exp_per_graph) + query_lls.unsqueeze(-1) + \
                      inner_log_weights.view(num_inner_graphs, 1, 1, 1)
                graph_info_gains[graph_idx] += tmp.logsumexp(dim=0).mean()
                graph_info_gains[graph_idx] = graph_info_gains[graph_idx] - data_mll

        return outer_log_weights.exp() @ graph_info_gains
