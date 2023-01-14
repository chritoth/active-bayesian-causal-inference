import itertools
import math
from typing import Tuple, Optional, List

import networkx as nx
import torch

from src.environments.environment import Experiment, gather_data, InterventionalDistributionsQuery
from src.models.graph_models import get_graph_key, get_parents
from src.models.mechanisms import Mechanism, GaussianProcess, GaussianRootNode


def get_mechanism_key(node, parents: List) -> str:
    parents = sorted(parents)
    key = str(node) + '<-' + ','.join([str(parent) for parent in parents])
    return key


def resolve_mechanism_key(key: str) -> Tuple[str, List[str]]:
    idx = key.find('<-')
    assert idx > 0, print('Invalid key: ' + key)
    node = key[:idx]
    parents = key[idx + 2:].split(',') if len(key) > idx + 2 else []
    return node, parents


class GaussianProcessModel:
    def __init__(self, node_labels: List[str], linear: bool = False):
        self.node_labels = sorted(list(set(node_labels)))
        self.linear = linear
        self.mechanisms = dict()
        self.mechanism_init_times = dict()
        self.mechanism_update_times = dict()
        self.topological_orders = dict()
        self.topological_orders_init_times = dict()
        self.prior_mll_cache = dict()
        self.posterior_mll_cache = dict()
        self.entropy_cache = dict()

    def get_parameters(self, keys: List[str] = None):
        keys = self.mechanisms.keys() if keys is None else keys
        param_lists = [list(self.mechanisms[key].parameters()) for key in keys]
        return list(itertools.chain(*param_lists))

    def init_mechanisms(self, graph: nx.DiGraph, init_time: int = 0):
        graph_key = get_graph_key(graph)
        if graph_key not in self.topological_orders and nx.is_directed_acyclic_graph(graph):
            self.topological_orders[graph_key] = list(nx.topological_sort(graph))
        if graph_key in self.topological_orders:
            self.topological_orders_init_times[graph_key] = init_time

        initialized_mechanisms = []
        for node in graph:
            parents = get_parents(node, graph)
            key = get_mechanism_key(node, parents)
            initialized_mechanisms.append(key)
            self.mechanism_init_times[key] = init_time
            if key not in self.mechanisms:
                self.mechanism_update_times[key] = 0
                self.mechanisms[key] = self.create_mechanism(len(parents))

        return initialized_mechanisms

    def discard_mechanisms(self, current_time: int, max_age: int):
        keys = [key for key, time in self.mechanism_init_times.items() if current_time - time > max_age]
        print(f'Discarding {len(keys)} old mechanisms...')
        for key in keys:
            del self.mechanisms[key]
            del self.mechanism_init_times[key]
            del self.mechanism_update_times[key]

        keys = [key for key, time in self.topological_orders_init_times.items() if current_time - time > max_age]
        print(f'Discarding {len(keys)} old topological orders...')
        for key in keys:
            del self.topological_orders[key]
            del self.topological_orders_init_times[key]

    def eval(self, keys: List[str] = None):
        keys = self.mechanisms.keys() if keys is None else keys
        for key in keys:
            self.mechanisms[key].eval()

    def train(self, keys: List[str] = None):
        keys = self.mechanisms.keys() if keys is None else keys
        for key in keys:
            self.mechanisms[key].train()

    def clear_prior_mll_cache(self, keys: List[str] = None):
        if keys is None:
            self.prior_mll_cache.clear()
        else:
            for key in keys:
                if key in self.prior_mll_cache:
                    del self.prior_mll_cache[key]

    def clear_posterior_mll_cache(self, keys: List[str] = None):
        if keys is None:
            self.posterior_mll_cache.clear()
        else:
            for key in keys:
                if key in self.posterior_mll_cache:
                    del self.posterior_mll_cache[key]

    def get_mechanism(self, node, graph: nx.DiGraph = None, parents: List[str] = None) -> Mechanism:
        assert graph is not None or parents is not None

        # get unique mechanism key
        parents = get_parents(node, graph) if parents is None else list(set(parents))
        key = get_mechanism_key(node, parents)

        # return mechanism if it already exists
        if key in self.mechanisms:
            return self.mechanisms[key]

        # if mechanism does not yet exists in the model, create a new mechanism
        num_parents = len(parents)
        self.mechanisms[key] = self.create_mechanism(num_parents)
        return self.mechanisms[key]

    def node_mll(self, experiments: List[Experiment], node: str, graph: nx.DiGraph, prior_mode=False,
                 use_cache=False, mode='joint', reduce=True) -> torch.Tensor:
        cache = self.prior_mll_cache if prior_mode else self.posterior_mll_cache
        parents = get_parents(node, graph)
        key = get_mechanism_key(node, parents)
        mll = torch.tensor(0.)
        if not use_cache or key not in cache:
            # gather data from the experiments
            inputs, targets = gather_data(experiments, node, parents=parents, mode=mode)
            # check if we have any data for this node
            if targets is not None:
                # compute log-likelihood
                mechanism = self.get_mechanism(node, parents=parents)
                try:
                    mll = mechanism.mll(inputs, targets, prior_mode, reduce=reduce)
                except Exception as e:
                    print(
                        f'Exception occured in GaussianProcessModel.mll() when computing MLL for mechanism {key} '
                        f'with prior mode {prior_mode} and use cache {use_cache}:')
                    print(e)
            # cache mll
            cache[key] = mll
        else:
            mll = cache[key]

        return mll

    def mll(self, experiments: List[Experiment], graph: nx.DiGraph, prior_mode=False, use_cache=False,
            mode='joint', reduce=True) -> torch.Tensor:
        mll = torch.tensor(0.)
        for node in self.node_labels:
            mll = mll + self.node_mll(experiments, node, graph, prior_mode, use_cache, mode=mode, reduce=reduce)
        return mll

    def expected_noise_entropy(self, interventions, graph: nx.DiGraph, use_cache=False) -> torch.Tensor:
        entropy = torch.tensor(0.)
        for node in self.node_labels:
            if node in interventions:
                continue

            parents = get_parents(node, graph)
            key = get_mechanism_key(node, parents)
            if not use_cache or key not in self.entropy_cache:
                # compute and cache entropy
                mechanism = self.get_mechanism(node, parents=parents)
                mechanism_entropy = mechanism.expected_noise_entropy()
                self.entropy_cache[key] = mechanism_entropy
            else:
                # take entropy from cache
                mechanism_entropy = self.entropy_cache[key]

            entropy += mechanism_entropy
        return entropy

    def get_num_mechanisms(self):
        return len(self.mechanisms)

    def mechanism_mlls(self, experiments: List[Experiment], keys: List[str] = None, prior_mode=False) -> torch.Tensor:
        # if no keys are given compute mlls for all mechanisms
        keys = self.mechanisms.keys() if keys is None else keys

        mlls = torch.tensor(0.)
        for key in keys:
            node, parents = resolve_mechanism_key(key)

            # gather data from the experiments
            inputs, targets = gather_data(experiments, node, parents=parents, mode='joint')

            # check if we have any data for this node
            if targets is None:
                continue

            # compute log-likelihood
            mechanism = self.mechanisms[key]
            try:
                mlls += mechanism.mll(inputs, targets, prior_mode) / targets.numel()
            except Exception as e:
                print(
                    f'Exception occured in GaussianProcessModel.mechanism_mlls() when computing MLL for mechanism '
                    f'{key} with prior  mode {prior_mode}:')
                print(e)
                if isinstance(mechanism, GaussianProcess):
                    print('Resampling GP hyperparameters...')
                    mechanism.init_hyperparams()
        return mlls

    def mechanism_log_hp_priors(self, keys: List[str] = None) -> torch.Tensor:
        # if no keys are given compute mlls for all mechanisms
        keys = self.mechanisms.keys() if keys is None else keys

        gps = [self.mechanisms[key] for key in keys if isinstance(self.mechanisms[key], GaussianProcess)]
        if not gps:
            return torch.tensor(0.)

        log_priors = torch.stack([gp.gp.hyperparam_log_prior() for gp in gps])
        return log_priors.sum()

    def create_mechanism(self, num_parents: int) -> Mechanism:
        return GaussianProcess(num_parents, linear=self.linear) if num_parents > 0 else GaussianRootNode()

    def set_data(self, experiments: List[Experiment], keys: List[str] = None):
        # if no keys are given set data for all mechanisms
        keys = self.mechanisms.keys() if keys is None else keys

        for key in keys:
            node, parents = resolve_mechanism_key(key)

            # gather data from the experiments
            inputs, targets = gather_data(experiments, node, parents=parents, mode='joint')

            # check if we have any data for this node
            if targets is None:
                continue

            # set GP data
            self.mechanisms[key].set_data(inputs, targets)

    def sample(self, interventions: dict, batch_size: int, num_batches: int, graph: nx.DiGraph) -> Experiment:
        data = dict()
        for node in self.topological_orders[get_graph_key(graph)]:
            # check if node is intervened upon
            if node in interventions:
                node_samples = torch.ones(num_batches, batch_size, 1) * interventions[node]
            else:
                # sample from mechanism
                parents = get_parents(node, graph)
                mechanism = self.get_mechanism(node, parents=parents)
                if not parents:
                    node_samples = mechanism.sample(torch.empty(num_batches, batch_size, 1))
                else:
                    x = torch.cat([data[parent] for parent in parents], dim=-1)
                    assert x.shape == (num_batches, batch_size, mechanism.in_size)
                    node_samples = mechanism.sample(x)

            # store samples
            data[node] = node_samples

        return Experiment(interventions, data)

    def interventional_mll(self, targets, node: str, interventions: dict, graph: nx.DiGraph, num_mc_samples=50,
                           reduce=True):
        assert targets.dim() == 2, print(f'Invalid shape {targets.shape}')
        num_batches, batch_size = targets.shape

        parents = get_parents(node, graph)
        mechanism = self.get_mechanism(node, parents=parents)

        if len(parents) == 0:
            # if we have a root note imll is simple
            mll = mechanism.mll(None, targets, prior_mode=False, reduce=False)
            assert mll.shape == (num_batches,), print(f'Invalid shape {mll.shape}!')
            return mll.sum() if reduce else mll

        # otherwise, do MC estimate via ancestral sampling
        samples = self.sample(interventions, batch_size, num_mc_samples, graph)
        # assemble inputs and targets
        inputs, _ = gather_data([samples], node, parents=parents, mode='independent_batches')
        inputs = inputs.unsqueeze(0).expand(num_batches, -1, -1, -1)
        assert inputs.shape == (num_batches, num_mc_samples, batch_size, len(parents))
        targets = targets.unsqueeze(1).expand(-1, num_mc_samples, batch_size)
        assert targets.shape == (num_batches, num_mc_samples, batch_size)
        # compute interventional mll
        mll = mechanism.mll(inputs, targets, prior_mode=False, reduce=False)
        assert mll.shape == (num_batches, num_mc_samples), print(f'Invalid shape {mll.shape}!')
        mll = mll.logsumexp(dim=1) - math.log(num_mc_samples)
        return mll.sum() if reduce else mll

    def sample_queries(self, queries: List[InterventionalDistributionsQuery], num_mc_queries: int,
                       num_batches_per_query: int, graph: nx.DiGraph):

        interventional_queries = [query.clone() for query in queries]
        with torch.no_grad():
            for query in interventional_queries:
                experiments = []
                for i in range(num_mc_queries):
                    interventions = query.sample_intervention()
                    experiments.append(self.sample(interventions, 1, num_batches_per_query, graph))

                query.set_sample_queries(experiments)

        return interventional_queries

    def query_log_probs(self, queries: List[InterventionalDistributionsQuery], graph: nx.DiGraph,
                        num_imll_mc_samples: int = 50):
        num_queries = len(queries)
        num_mc_queries = len(queries[0].sample_queries)
        num_batches_per_query = queries[0].sample_queries[0].num_batches

        query_lls = torch.zeros(num_mc_queries, num_queries, num_batches_per_query)
        for i in range(num_mc_queries):
            for query_idx, query in enumerate(queries):
                query_node = query.query_nodes[0]  # ToDo: supports only single query node!!!
                targets = query.sample_queries[i].data[query_node].squeeze(-1)
                imll = self.interventional_mll(targets, query_node, query.sample_queries[i].interventions, graph,
                                               num_mc_samples=num_imll_mc_samples, reduce=False)
                query_lls[i, query_idx] = imll

        return query_lls.sum(dim=1)

    def update_gp_hyperparameters(self, update_time: int, experiments: List[Experiment], set_data=False,
                                  mechanisms: Optional[List[str]] = None):
        assert 0 < update_time <= len(experiments), print(f'Cannot update on {update_time}/{len(experiments)} '
                                                          f'experiments!')
        if mechanisms is None:
            mechanisms = list(self.mechanisms.keys())

        keys = [key for key in mechanisms if self.mechanism_update_times[key] != update_time]

        if keys:
            print(f'Updating {len(keys)} GP\'s hyperparams with first {update_time} experiments.')
            max_update_size = 500
            num_full_batches = len(keys) // max_update_size
            key_batches = [keys[i * max_update_size:(i + 1) * max_update_size] for i in range(num_full_batches)]
            if len(keys) % max_update_size > 0:
                tmp = keys[max_update_size * num_full_batches:]
                key_batches.append(tmp)

            for batch in key_batches:
                self.set_data(experiments[:update_time], batch)
                self.optimize_gp_hyperparams(experiments[:update_time], batch)
                self.clear_prior_mll_cache(batch)
                self.clear_posterior_mll_cache(batch)
                for key in batch:
                    self.mechanism_update_times[key] = update_time

        # set training data for mechanisms
        if set_data:
            self.set_data(experiments, mechanisms)

        # put mechanisms into eval mode
        self.eval(mechanisms)

    def optimize_gp_hyperparams(self, experiments: List[Experiment], keys: List[str] = None, num_steps: int = 70,
                                log_interval: int = 0):
        keys = self.mechanisms.keys() if keys is None else keys
        params = self.get_parameters(keys)
        if not params:
            return

        optimizer = torch.optim.RMSprop(params, lr=2e-2)

        losses = []
        self.train(keys)
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = -self.mechanism_mlls(experiments, keys)
            loss -= self.mechanism_log_hp_priors(keys)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if i > 5 and torch.tensor(losses[-4:-1]).mean() - torch.tensor(losses[-5:-2]).mean() < 1e-3:
                # print(f'\nStopping GP parameter update early after improvement stagnates...')
                break

            if log_interval < 1:
                continue

            print(f'Step {i + 1} of {num_steps}, negative MLL is {loss.item()}...',
                  end='' if i % log_interval == 0 else '\r', flush=True)

        return losses

    def submodel(self, graphs):
        mechanisms_keys = {get_mechanism_key(node, get_parents(node, graph)) for graph in graphs for node in graph}
        submodel = self.__class__(self.node_labels)
        submodel.mechanisms = {key: self.mechanisms[key] for key in mechanisms_keys}
        submodel.topological_orders = self.topological_orders
        return submodel

    def param_dict(self):
        mechanism_param_dict = {key: m.param_dict() for key, m in self.mechanisms.items()}
        params = {'node_labels': self.node_labels,
                  'linear': self.linear,
                  'mechanism_init_times': self.mechanism_init_times,
                  'mechanism_update_times': self.mechanism_update_times,
                  'topological_orders': self.topological_orders,
                  'mechanism_param_dict': mechanism_param_dict}
        return params

    def load_param_dict(self, param_dict):
        self.node_labels = param_dict['node_labels']
        self.linear = param_dict['linear']
        self.mechanism_init_times = param_dict['mechanism_init_times']
        self.mechanism_update_times = param_dict['mechanism_update_times']
        self.topological_orders = param_dict['topological_orders']
        for key, d in param_dict['mechanism_param_dict'].items():
            self.mechanisms[key] = self.create_mechanism(d['in_size'])
            self.mechanisms[key].load_param_dict(d)
