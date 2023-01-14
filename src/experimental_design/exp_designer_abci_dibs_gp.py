from typing import Dict, Tuple, List, Optional

import networkx as nx
import torch.optim

from src.environments.environment import Experiment, InterventionalDistributionsQuery
from src.experimental_design.exp_designer_base import ExpDesignerBase
from src.models.gp_model import GaussianProcessModel
from src.models.graph_models import get_graph_key


class ExpDesignerABCIDiBSGP(ExpDesignerBase):
    model: Optional[GaussianProcessModel]

    def __init__(self, intervention_bounds: Dict[str, Tuple[float, float]], opt_strategy: str = 'gp-ucb',
                 distributed=False):
        super().__init__(intervention_bounds, opt_strategy, distributed)
        self.model = None

    def init_design_process(self, args: dict):
        self.model = args['mechanism_model']

        if args['policy'] == 'scm-info-gain':
            def utility(interventions: dict):
                self.model.entropy_cache.clear()
                return self.scm_info_gain(interventions,
                                          args['inner_mc_graphs'],
                                          args['log_inner_graph_weights'],
                                          args['log_inner_particle_weights'],
                                          args['outer_mc_graphs'],
                                          args['outer_graph_weights'],
                                          args['outer_particle_weights'],
                                          args['batch_size'],
                                          args['num_exp_per_graph'])
        elif args['policy'] == 'graph-info-gain':
            def utility(interventions: dict):
                return self.graph_info_gain(interventions,
                                            args['inner_mc_graphs'],
                                            args['log_inner_graph_weights'],
                                            args['log_inner_particle_weights'],
                                            args['outer_mc_graphs'],
                                            args['outer_graph_weights'],
                                            args['outer_particle_weights'],
                                            args['batch_size'],
                                            args['num_exp_per_graph'])
        elif args['policy'] == 'intervention-info-gain':
            def utility(interventions: dict):
                return self.intervention_info_gain(interventions,
                                                   args['experiments'],
                                                   args['interventional_queries'],
                                                   args['inner_mc_graphs'],
                                                   args['log_inner_graph_weights'],
                                                   args['log_inner_particle_weights'],
                                                   args['outer_mc_graphs'],
                                                   args['outer_graph_weights'],
                                                   args['outer_particle_weights'],
                                                   args['num_mc_queries'],
                                                   args['num_batches_per_query'],
                                                   args['batch_size'],
                                                   args['num_exp_per_graph'])
        else:
            assert False, 'Invalid policy ' + args['policy'] + '!'

        self.utility = utility

    def compute_graph_posterior_mlls(self, experiments: List[Experiment], graphs: List[List[nx.DiGraph]],
                                     mode='independent_batches', reduce=True) -> torch.Tensor:
        num_particles = len(graphs)
        num_mc_graphs = len(graphs[0])
        graph_mlls = [self.model.mll(experiments, graph, prior_mode=False, use_cache=True, mode=mode, reduce=reduce)
                      for i in range(num_particles) for graph in graphs[i]]
        graph_mlls = torch.stack(graph_mlls).view(num_particles, num_mc_graphs, *graph_mlls[0].shape)
        return graph_mlls

    def graph_info_gain(self, interventions: dict,
                        inner_mc_graphs: List[List[nx.DiGraph]],
                        log_inner_graph_weights: torch.Tensor,
                        log_inner_particle_weights: torch.Tensor,
                        outer_mc_graphs: List[List[nx.DiGraph]],
                        outer_graph_weights: torch.Tensor,
                        outer_particle_weights: torch.Tensor,
                        batch_size: int = 1,
                        num_exp_per_graph: int = 1):
        num_particles = len(outer_mc_graphs)
        num_outer_mc_graphs = len(outer_mc_graphs[0])

        with torch.no_grad():
            graph_info_gains = torch.zeros(num_particles, num_outer_mc_graphs)
            for particle_idx, graphs in enumerate(outer_mc_graphs):
                for graph_idx, graph in enumerate(graphs):
                    # check if graph is acyclic
                    if get_graph_key(graph) not in self.model.topological_orders:
                        print(f'Worker {self.worker_id}: could not sample from cyclic graph.')
                        continue

                    simulated_experiments = [self.model.sample(interventions, batch_size, num_exp_per_graph, graph)]

                    self.model.clear_posterior_mll_cache()
                    outer_posterior_mll = self.model.mll(simulated_experiments, graph, prior_mode=False,
                                                         use_cache=True, mode='independent_batches', reduce=False)
                    outer_posterior_mll = outer_posterior_mll.sum()

                    # compute log p(D|D_old)
                    inner_posterior_mlls = self.compute_graph_posterior_mlls(simulated_experiments,
                                                                             inner_mc_graphs,
                                                                             reduce=False)
                    particle_posterior_mlls = (inner_posterior_mlls +
                                               log_inner_graph_weights.unsqueeze(-1)).logsumexp(dim=1)
                    data_mll = (log_inner_particle_weights.unsqueeze(-1) + particle_posterior_mlls).logsumexp(dim=0)
                    data_mll = data_mll.sum()

                    graph_info_gains[particle_idx, graph_idx] += outer_posterior_mll - data_mll

            # compute info gain
            graph_info_gains /= num_exp_per_graph
            expected_info_gain = outer_particle_weights @ (outer_graph_weights * graph_info_gains).sum(dim=1)
        return expected_info_gain

    def scm_info_gain(self, interventions: dict,
                      inner_mc_graphs: List[List[nx.DiGraph]],
                      log_inner_graph_weights: torch.Tensor,
                      log_inner_particle_weights: torch.Tensor,
                      outer_mc_graphs: List[List[nx.DiGraph]],
                      outer_graph_weights: torch.Tensor,
                      outer_particle_weights: torch.Tensor,
                      batch_size: int = 1,
                      num_exp_per_graph: int = 1):
        num_particles = len(outer_mc_graphs)
        num_outer_mc_graphs = len(outer_mc_graphs[0])

        with torch.no_grad():
            graph_info_gains = torch.zeros(num_particles, num_outer_mc_graphs)
            for particle_idx, graphs in enumerate(outer_mc_graphs):
                for graph_idx, graph in enumerate(graphs):
                    # check if graph is acyclic
                    if get_graph_key(graph) not in self.model.topological_orders:
                        print(f'Worker {self.worker_id}: could not sample from cyclic graph.')
                        continue

                    simulated_experiments = [self.model.sample(interventions, batch_size, num_exp_per_graph, graph)]

                    self.model.clear_posterior_mll_cache()
                    inner_posterior_mlls = self.compute_graph_posterior_mlls(simulated_experiments,
                                                                             inner_mc_graphs,
                                                                             reduce=False)

                    # compute log p(D|D_old)
                    particle_posterior_mlls = (inner_posterior_mlls +
                                               log_inner_graph_weights.unsqueeze(-1)).logsumexp(dim=1)
                    data_mll = (log_inner_particle_weights.unsqueeze(-1) + particle_posterior_mlls).logsumexp(dim=0)
                    data_mll = data_mll.sum()

                    entropy = self.model.expected_noise_entropy(interventions, outer_mc_graphs[particle_idx][graph_idx],
                                                                use_cache=True)
                    graph_info_gains[particle_idx, graph_idx] += -entropy * batch_size - data_mll / num_exp_per_graph

            # compute info gain
            expected_info_gain = outer_particle_weights @ (outer_graph_weights * graph_info_gains).sum(dim=1)
            return expected_info_gain

    def intervention_info_gain(self, interventions: dict,
                               experiments: List[Experiment],
                               interventional_queries: List[InterventionalDistributionsQuery],
                               inner_mc_graphs: List[List[nx.DiGraph]],
                               log_inner_graph_weights: torch.Tensor,
                               log_inner_particle_weights: torch.Tensor,
                               outer_mc_graphs: List[List[nx.DiGraph]],
                               outer_graph_weights: torch.Tensor,
                               outer_particle_weights: torch.Tensor,
                               num_mc_queries: int = 1,
                               num_batches_per_query: int = 1,
                               batch_size: int = 1,
                               num_exp_per_graph: int = 1):
        # get mc graphs and compute log weights
        num_particles = len(outer_mc_graphs)
        num_outer_graphs = len(outer_mc_graphs[0])
        num_inner_graphs = len(inner_mc_graphs[0])

        # check if all inner graphs are acyclic and zero the weights of the cyclic ones
        for particle_idx in range(num_particles):
            for graph_idx, graph in enumerate(inner_mc_graphs[particle_idx]):
                if get_graph_key(graph) not in self.model.topological_orders:
                    print(f'Worker {self.worker_id}: cannot evaluate query ll for cyclic graph.')
                    log_inner_graph_weights[particle_idx, graph_idx] = torch.tensor(-1e8)

        # compute the info gain
        with torch.no_grad():
            graph_info_gains = torch.zeros(num_particles, num_outer_graphs)
            for particle_idx in range(num_particles):
                for graph_idx, graph in enumerate(outer_mc_graphs[particle_idx]):
                    # check if graph is acyclic
                    if get_graph_key(graph) not in self.model.topological_orders:
                        print(f'Worker {self.worker_id}: could not sample from cyclic graph.')
                        continue

                    # simulate experiments and compute data mll
                    self.model.set_data(experiments)
                    self.model.clear_posterior_mll_cache()
                    simulated_experiments = [self.model.sample(interventions, batch_size, num_exp_per_graph, graph)]
                    posterior_mlls = self.compute_graph_posterior_mlls(simulated_experiments, inner_mc_graphs,
                                                                       reduce=False)
                    assert posterior_mlls.shape == (num_particles, num_inner_graphs, num_exp_per_graph)

                    particle_posterior_mlls = (posterior_mlls + log_inner_graph_weights.unsqueeze(-1)).logsumexp(dim=1)
                    data_mll = (log_inner_particle_weights.unsqueeze(-1) + particle_posterior_mlls).logsumexp(dim=0)
                    data_mll = data_mll.mean()

                    # simulate queries
                    self.model.set_data(experiments + simulated_experiments)
                    self.model.clear_posterior_mll_cache()

                    # simulate queries & compute query lls
                    mc_queries = self.model.sample_queries(interventional_queries, num_mc_queries,
                                                           num_batches_per_query, graph)

                    query_lls = []
                    for i in range(num_particles):
                        for g in inner_mc_graphs[i]:
                            # check if graph is acyclic
                            if get_graph_key(g) in self.model.topological_orders:
                                query_lls.append(self.model.query_log_probs(mc_queries, g))
                            else:
                                query_lls.append(torch.tensor(0.))

                    query_lls = torch.stack(query_lls).view(num_particles, num_inner_graphs, num_mc_queries,
                                                            num_batches_per_query)

                    # compute and update per graph info gain
                    tmp = posterior_mlls.view(num_particles, num_inner_graphs, 1, 1, num_exp_per_graph) + \
                          query_lls.unsqueeze(-1) + \
                          log_inner_graph_weights.view(num_particles, num_inner_graphs, 1, 1, 1)
                    tmp = tmp.logsumexp(dim=1)
                    tmp = (tmp + log_inner_particle_weights.view(num_particles, 1, 1, 1)).logsumexp(dim=0)
                    graph_info_gains[graph_idx] += tmp.mean()

                    graph_info_gains[graph_idx] = graph_info_gains[graph_idx] - data_mll

        expected_info_gain = outer_particle_weights @ (outer_graph_weights * graph_info_gains).sum(dim=1)
        return expected_info_gain
