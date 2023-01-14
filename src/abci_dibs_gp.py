from typing import Tuple, Callable

import torch.optim
from torch.nn.functional import log_softmax

from src.abci_base import ABCIBase
from src.environments.environment import *
from src.experimental_design.exp_designer_abci_dibs_gp import ExpDesignerABCIDiBSGP
from src.models.gp_model import GaussianProcessModel
from src.models.graph_models import DiBSModel, get_graph_key
from src.utils.metrics import shd, auroc, auprc


class ABCIDiBSGP(ABCIBase):
    policies = {'observational', 'random', 'random-fixed-value', 'graph-info-gain', 'scm-info-gain',
                'intervention-info-gain'}

    def __init__(self, env: Environment, policy: str, num_particles: int = 5,
                 num_mc_graphs: int = 40, embedding_size: int = None, num_workers: int = 1, dibs_plus=True,
                 linear: bool = False):
        assert policy in self.policies, print(f'Invalid policy {policy}!')
        super().__init__(env, policy, num_workers)

        # store params
        embedding_size = env.num_nodes if embedding_size is None else embedding_size
        self.num_particles = num_particles
        self.num_mc_graphs = num_mc_graphs
        self.embedding_size = embedding_size
        self.dibs_plus = dibs_plus
        self.linear = linear

        # init models
        self.graph_model = DiBSModel(env.node_labels, embedding_size, num_particles)
        self.mechanism_model = GaussianProcessModel(env.node_labels, linear=linear)

        # init mc graphs
        self.mc_graphs = self.mc_adj_mats = None

    def experiment_designer_factory(self):
        distributed = self.num_workers > 1
        return ExpDesignerABCIDiBSGP(self.env.intervention_bounds, opt_strategy='gp-ucb', distributed=distributed)

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
            else:
                if self.policy in {'graph-info-gain', 'scm-info-gain'}:
                    # sample mc graphs
                    outer_mc_graphs, _ = self.sample_mc_graphs(set_data=True, num_graphs=5, only_dags=False)
                    inner_mc_graphs, _ = self.sample_mc_graphs(set_data=True, num_graphs=30, only_dags=False)
                    with torch.no_grad():
                        log_inner_graph_weights, log_inner_particle_weights = self.compute_importance_weights(
                            inner_mc_graphs, use_cache=True, log_weights=True)
                        outer_graph_weights, outer_particle_weights = self.compute_importance_weights(outer_mc_graphs,
                                                                                                      use_cache=True)

                    graphs = [g for glist in inner_mc_graphs for g in glist]
                    graphs += [g for glist in outer_mc_graphs for g in glist]
                    args = {'mechanism_model': self.mechanism_model.submodel(graphs),
                            'inner_mc_graphs': inner_mc_graphs,
                            'log_inner_graph_weights': log_inner_graph_weights,
                            'log_inner_particle_weights': log_inner_particle_weights,
                            'outer_mc_graphs': outer_mc_graphs,
                            'outer_graph_weights': outer_graph_weights,
                            'outer_particle_weights': outer_particle_weights,
                            'batch_size': batch_size,
                            'num_exp_per_graph': 100,
                            'policy': self.policy}

                elif self.policy in {'intervention-info-gain'}:
                    self.mechanism_model.discard_mechanisms(len(self.experiments), -1)

                    # sample mc graphs
                    outer_mc_graphs, _ = self.sample_mc_graphs(set_data=True, num_graphs=3, only_dags=False)
                    inner_mc_graphs, _ = self.sample_mc_graphs(set_data=True, num_graphs=9, only_dags=False)
                    with torch.no_grad():
                        # compute importance weights
                        log_inner_graph_weights, log_inner_particle_weights = self.compute_importance_weights(
                            inner_mc_graphs, use_cache=True, log_weights=True)
                        outer_graph_weights, outer_particle_weights = self.compute_importance_weights(outer_mc_graphs,
                                                                                                      use_cache=True)

                    graphs = [g for glist in inner_mc_graphs for g in glist]
                    graphs += [g for glist in outer_mc_graphs for g in glist]
                    args = {'mechanism_model': self.mechanism_model.submodel(graphs),
                            'experiments': self.experiments,
                            'interventional_queries': self.env.interventional_queries,
                            'inner_mc_graphs': inner_mc_graphs,
                            'log_inner_graph_weights': log_inner_graph_weights,
                            'log_inner_particle_weights': log_inner_particle_weights,
                            'outer_mc_graphs': outer_mc_graphs,
                            'outer_graph_weights': outer_graph_weights,
                            'outer_particle_weights': outer_particle_weights,
                            'num_mc_queries': 5,
                            'num_batches_per_query': 3,
                            'batch_size': batch_size,
                            'num_exp_per_graph': 50,
                            'policy': self.policy}
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

            # resample particles
            num_experiments_conducted = len(self.experiments)
            resampling_interval = 1 if num_experiments_conducted < 6 else (3 if num_experiments_conducted < 10 else 5)
            if num_experiments_conducted > 0 and num_experiments_conducted % resampling_interval == 0:
                self.resample_particles(use_cache=True)
                # clear mechs & caches: many different mechs after resampling
                self.mechanism_model.discard_mechanisms(num_experiments_conducted, -1)

            # perform experiment
            num_samples = batch_size
            if num_experiments_conducted == 0 and num_initial_obs_samples > 0:
                num_samples = num_initial_obs_samples
            self.experiments.append(self.env.sample(interventions, num_samples))

            # clear caches
            self.mechanism_model.clear_prior_mll_cache()
            self.mechanism_model.clear_posterior_mll_cache()
            self.mechanism_model.entropy_cache.clear()

            # update latent particles
            # num_steps = 100 if len(self.experiments) < 10 else 200
            print(f'Updating latent particles via SVGD...', flush=True)
            num_steps = 500
            losses = self.update_latent_particles(num_steps)
            self.loss_list.extend(losses)

            # discard old mechanisms
            self.mechanism_model.discard_mechanisms(len(self.experiments), max_age=0)
            print(f'There are currently {self.mechanism_model.get_num_mechanisms()} unique mechanisms in our model...')

            print(f'Logging evaluation stats...', flush=True)
            self.mc_graphs, self.mc_adj_mats = self.sample_mc_graphs(set_data=True)

            # record graph posterior entropy
            self.graph_entropy_list.append(torch.tensor(-1.))

            # record expected SHD
            with torch.no_grad():
                eshd = self.graph_posterior_expectation(lambda g: shd(self.env.graph, g))
            self.eshd_list.append(eshd)

            # record env graph LL
            graph_ll = self.compute_graph_log_posterior(self.env.graph)
            self.graph_ll_list.append(graph_ll)

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
                posterior_edge_probs = self.compute_posterior_edge_probs(use_cache=True)
                true_adj_mat = self.graph_model.graph_to_adj_mat(self.env.graph)

            self.auroc_list.append(auroc(posterior_edge_probs, true_adj_mat))
            self.auprc_list.append(auprc(posterior_edge_probs, true_adj_mat))

            # record query KLD
            if self.env.interventional_queries is not None:
                self.mc_graphs, self.mc_adj_mats = self.sample_mc_graphs(set_data=True, num_graphs=5, only_dags=False)

                def test_query_ll(graph):
                    # check if graph is acyclic
                    if get_graph_key(graph) not in self.mechanism_model.topological_orders:
                        print(f'Cannot evaluate test query ll for cyclic graph.')
                        num_mc_queries = len(self.env.interventional_queries[0].sample_queries)
                        num_batches_per_query = self.env.interventional_queries[0].sample_queries[0].num_batches
                        return torch.ones(num_mc_queries, num_batches_per_query) * 1e-8

                    query_lls = self.mechanism_model.query_log_probs(self.env.interventional_queries, graph, 200)
                    return query_lls

                self.mechanism_model.clear_posterior_mll_cache()
                with torch.no_grad():
                    ll = self.graph_posterior_expectation(test_query_ll, logspace=True).mean()
                self.query_kld_list.append(self.env.query_ll - ll)

            if outdir is not None and 0 < epoch < num_experiments - 1 and (epoch + 1) % checkpoint_interval == 0:
                model = 'abci-dibs-gp-linear' if self.linear else 'abci-dibs-gp'
                outpath = outdir + model + '-' + self.policy + '-' + self.env.name + f'-{job_id}-exp-{epoch + 1}.pth'
                self.save(outpath)

            if log_interval > 0 and epoch % log_interval == 0:
                print(f'Experiment {epoch + 1}/{num_experiments}, ESHD is {eshd.item()}', flush=True)

    def resample_particles(self, threshold: float = 1e-2, use_cache: bool = False):
        mc_graphs, _ = self.sample_mc_graphs()
        num_particles = len(mc_graphs)

        max_particles_to_keep = math.ceil(num_particles / 4.)
        with torch.no_grad():
            _, particle_weights = self.compute_importance_weights(mc_graphs, use_cache=use_cache, dibs_plus=True)

            particle_idc = particle_weights.argsort(descending=True).numpy()
            num_kept = 0
            resampled_particles = []
            for i in particle_idc:
                if num_kept >= max_particles_to_keep or particle_weights[i] < threshold:
                    self.graph_model.particles[i] = self.graph_model.sample_initial_particles(1).squeeze(0)
                    resampled_particles.append(i)
                else:
                    num_kept += 1

        print(f'Resampling particles {resampled_particles} according to weights {particle_weights.squeeze()} (kept '
              f'{num_kept}/{max_particles_to_keep}')

    def update_latent_particles(self, num_steps: int = 100, log_interval: int = 25):
        optimizer = torch.optim.Adam([self.graph_model.particles], lr=1e-1)
        # alphas = 1. + 1e-2 * torch.arange(1, num_steps + 1).numpy()
        alphas = torch.ones(num_steps).numpy()
        betas = 1. + 25e-2 * torch.arange(1, num_steps + 1).numpy()
        # betas = 50. * torch.ones(num_steps).numpy()

        losses = []
        for i in range(num_steps):
            self.mc_graphs, self.mc_adj_mats = self.sample_mc_graphs(alphas[i])

            optimizer.zero_grad()
            log_posterior_grads, unnormalized_log_posterior = self.estimate_score_function(alphas[i], betas[i])

            bandwidth = self.graph_model.embedding_size * self.graph_model.num_nodes * 2.
            # bandwidth = 10.
            particle_similarities = self.graph_model.particle_similarities(bandwidth=bandwidth)
            sim_grads = torch.autograd.grad(particle_similarities.sum(), self.graph_model.particles)[0]
            particle_grads = torch.einsum('ab,bdef->adef', particle_similarities.detach(),
                                          log_posterior_grads) - sim_grads

            self.graph_model.particles.grad = -particle_grads / self.graph_model.num_particles

            optimizer.step()
            losses.append(-unnormalized_log_posterior.item())

            if i > 50:
                change = (torch.tensor(losses[-4:-1]).mean() - torch.tensor(losses[-5:-2]).mean()).abs()
                if change < 1e-3:
                    print(f'Stopping particle updates early in iteration {i} after improvement stagnates...')
                    break

            if log_interval > 0 and i % log_interval == 0:
                print(f'Step {i + 1} of {num_steps}, negative log posterior is {-unnormalized_log_posterior.item()}...',
                      flush=True)

        return losses

    def estimate_score_function(self, alpha: float = 1., beta: float = 1.) -> Tuple[torch.Tensor, torch.Tensor]:
        num_particles, num_mc_graphs = self.mc_adj_mats.shape[0:2]

        # compute log prior p(Z)
        log_prior = self.graph_model.unnormalized_log_prior(alpha, beta).sum()

        # compute graph weights with baseline for variance reduction (p(D|G) - b) / p(D|Z)
        with torch.no_grad():
            graph_mlls = self.compute_graph_mlls(self.mc_graphs, use_cache=True)
            log_normalization = graph_mlls.logsumexp(dim=1)
            particle_mlls = log_normalization - math.log(num_mc_graphs)
            graph_weights = (graph_mlls - log_normalization.unsqueeze(1)).exp()

        baseline = torch.ones(num_particles, 1) / num_mc_graphs

        # compute log generative probabilities p(G|Z)
        log_generative_probs = self.graph_model.log_generative_prob(self.mc_adj_mats, alpha)
        tmp = log_prior + ((graph_weights - baseline) * log_generative_probs).sum()
        score_func = torch.autograd.grad(tmp, self.graph_model.particles)[0]

        unnormalized_log_posterior = (log_prior + particle_mlls.sum()) / num_particles
        return score_func, unnormalized_log_posterior

    def sample_mc_graphs(self, alpha: float = 1., set_data=False, num_graphs: int = None, only_dags=False):
        num_graphs = self.num_mc_graphs if num_graphs is None else num_graphs

        with torch.no_grad():
            mc_graphs, mc_adj_mats = self.graph_model.sample_graphs(num_graphs, alpha)
            if only_dags:
                self.graph_model.dagify_graphs(mc_graphs, mc_adj_mats)

        self.init_graph_mechanisms(mc_graphs, set_data=set_data)

        return mc_graphs, mc_adj_mats

    def init_graph_mechanisms(self, graphs: List[List[nx.DiGraph]], set_data=False):
        # initialize mechanisms
        num_particles = len(graphs)
        time = len(self.experiments)
        initialized_mechanisms = set()
        for i in range(num_particles):
            for graph in graphs[i]:
                keys = self.mechanism_model.init_mechanisms(graph, init_time=time)
                initialized_mechanisms.update(keys)
        initialized_mechanisms = list(initialized_mechanisms)

        # update mechanism hyperparameters
        hyperparam_update_interval = 1 if time < 6 else (3 if time < 10 else 5)
        if hyperparam_update_interval > 0:
            update_time = 1 if time <= 1 else (time // hyperparam_update_interval) * hyperparam_update_interval

            self.mechanism_model.update_gp_hyperparameters(update_time, self.experiments, set_data,
                                                           initialized_mechanisms)

    def compute_graph_mlls(self, graphs: List[List[nx.DiGraph]], experiments: List[Experiment] = None, prior_mode=True,
                           use_cache=True):
        num_particles = len(graphs)
        num_mc_graphs = len(graphs[0])
        experiments = self.experiments if experiments is None else experiments
        graph_mlls = [self.mechanism_model.mll(experiments, graph, prior_mode=prior_mode, use_cache=use_cache) for i in
                      range(num_particles) for graph in graphs[i]]
        graph_mlls = torch.stack(graph_mlls).view(num_particles, num_mc_graphs)
        return graph_mlls

    def graph_posterior_expectation(self, func: Callable[[nx.DiGraph], torch.Tensor], use_cache=True, logspace=False):
        num_particles, num_mc_graphs = self.mc_adj_mats.shape[0:2]

        # compute function values
        func_values = [func(graph) for i in range(num_particles) for graph in self.mc_graphs[i]]
        func_output_shape = func_values[0].shape
        func_output_dim = len(func_output_shape)
        func_values = torch.stack(func_values).view(num_particles, num_mc_graphs, *func_output_shape)

        # compute expectation
        if logspace:
            log_graph_weights, log_particle_weights = self.compute_importance_weights(self.mc_graphs,
                                                                                      use_cache=use_cache,
                                                                                      log_weights=True)
            log_graph_weights = log_graph_weights.view(num_particles, num_mc_graphs, *([1] * func_output_dim))
            log_particle_weights = log_particle_weights.view(num_particles, *([1] * func_output_dim))
            expected_value = (log_graph_weights + func_values).logsumexp(dim=1)
            expected_value = (log_particle_weights + expected_value).logsumexp(dim=0)
            return expected_value

        graph_weights, particle_weights = self.compute_importance_weights(self.mc_graphs, use_cache=use_cache)
        graph_weights = graph_weights.view(num_particles, num_mc_graphs, *([1] * func_output_dim))
        particle_weights = particle_weights.view(num_particles, *([1] * func_output_dim))

        expected_value = particle_weights @ (graph_weights * func_values).sum(dim=1)
        return expected_value

    def compute_posterior_edge_probs(self, use_cache=True):
        num_particles, num_mc_graphs = self.mc_adj_mats.shape[0:2]

        # compute expectation
        graph_weights, particle_weights = self.compute_importance_weights(self.mc_graphs, use_cache=use_cache)

        posterior_edge_probs = torch.zeros(self.env.num_nodes, self.env.num_nodes)
        for i in range(self.env.num_nodes):
            for j in range(self.env.num_nodes):
                particle_edge_probs = torch.zeros(num_particles)
                for particle_idx in range(num_particles):
                    particle_edge_probs[particle_idx] = sum([graph_weights[particle_idx, graph_idx] for graph_idx
                                                             in range(num_mc_graphs) if
                                                             self.mc_adj_mats[particle_idx, graph_idx, i, j].bool()])

                posterior_edge_probs[i, j] = particle_weights @ particle_edge_probs

        return posterior_edge_probs

    def compute_graph_log_posterior(self, graph: nx.DiGraph, alpha: float = 1.):
        num_particles = len(self.mc_graphs)
        num_mc_graphs = len(self.mc_graphs[0])

        self.init_graph_mechanisms([[graph]])
        with torch.no_grad():
            graph_mlls = self.compute_graph_mlls(self.mc_graphs)
            log_prior = self.graph_model.unnormalized_log_prior(beta=50.)
            log_normalization = graph_mlls.logsumexp(dim=1)
            particle_mlls = log_normalization - math.log(num_mc_graphs)
            log_particle_weights = log_softmax(log_prior + particle_mlls, dim=0) if self.dibs_plus else \
                -torch.tensor(num_particles).log()

            adj_mat = self.graph_model.graph_to_adj_mat(graph).expand(num_particles, 1, -1, -1)
            log_generative_probs = self.graph_model.log_generative_prob(adj_mat, alpha).squeeze()
            tmp = (log_generative_probs - particle_mlls + log_particle_weights).logsumexp(dim=0)

            log_graph_posterior = tmp + self.mechanism_model.mll(self.experiments, graph, prior_mode=True)

        return log_graph_posterior

    def compute_importance_weights(self, mc_graphs, use_cache=False, beta=50., log_weights: bool = False,
                                   dibs_plus: bool = None):
        if dibs_plus is None:
            dibs_plus = self.dibs_plus

        num_particles = len(mc_graphs)
        num_mc_graphs = len(mc_graphs[0])
        graph_mlls = self.compute_graph_mlls(mc_graphs, use_cache=use_cache)
        log_normalization = graph_mlls.logsumexp(dim=1)
        log_graph_weights = (graph_mlls - log_normalization.unsqueeze(1))
        if dibs_plus:
            log_particle_prior = self.graph_model.unnormalized_log_prior(beta=beta)
            particle_mlls = log_normalization - math.log(num_mc_graphs)
            log_particle_weights = log_softmax(log_particle_prior + particle_mlls, dim=0)
        else:
            log_particle_weights = -torch.tensor(num_particles).log() * torch.ones(num_particles)

        if log_weights:
            return log_graph_weights, log_particle_weights
        return log_graph_weights.exp(), log_particle_weights.exp()

    def param_dict(self):
        params = super().param_dict()
        params.update({'num_particles': self.num_particles,
                       'num_mc_graphs': self.num_mc_graphs,
                       'embedding_size': self.embedding_size,
                       'dibs_plus': self.dibs_plus,
                       'linear': self.linear,
                       'mechanism_model_params': self.mechanism_model.param_dict(),
                       'graph_model_params': self.graph_model.param_dict()})
        return params

    def load_param_dict(self, param_dict):
        super().load_param_dict(param_dict)
        self.num_particles = param_dict['num_particles']
        self.num_mc_graphs = param_dict['num_mc_graphs']
        self.embedding_size = param_dict['embedding_size']
        self.dibs_plus = param_dict['dibs_plus']
        self.linear = param_dict['linear']
        self.mechanism_model.load_param_dict(param_dict['mechanism_model_params'])
        self.graph_model.load_param_dict(param_dict['graph_model_params'])

    def save(self, path):
        torch.save(self.param_dict(), path)

    @classmethod
    def load(cls, path, num_workers: int = 1):
        param_dict = torch.load(path)

        env_param_dict = param_dict['env_param_dict']
        env = Environment(env_param_dict['num_nodes'], mechanism_model=None, num_test_samples_per_intervention=0,
                          num_test_queries=0, graph=env_param_dict['graph'])
        env.load_param_dict(env_param_dict)

        abci = ABCIDiBSGP(env, param_dict['policy'], param_dict['num_particles'],
                          param_dict['num_mc_graphs'], param_dict['embedding_size'], num_workers,
                          param_dict['dibs_plus'], param_dict['linear'])
        abci.load_param_dict(param_dict)
        return abci
