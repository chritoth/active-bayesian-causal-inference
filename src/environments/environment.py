import random
import string
from typing import Dict, List, Set, Optional

import networkx as nx

from src.environments.experiment import Experiment, gather_data
from src.models.graph_models import get_parents
from src.models.mechanisms import *


class InterventionalDistributionsQuery:
    def __init__(self, query_nodes: List[str], intervention_targets: Dict[str, dist.Distribution],
                 sample_queries: List[Experiment] = None):
        self.query_nodes = query_nodes
        self.intervention_targets = intervention_targets
        self.sample_queries = sample_queries

    def sample_intervention(self):
        return {target: distr.sample() for target, distr in self.intervention_targets.items()}

    def set_sample_queries(self, sample_queries: List[Experiment]):
        self.sample_queries = sample_queries

    def clone(self):
        return InterventionalDistributionsQuery(self.query_nodes, self.intervention_targets)

    def param_dict(self):
        params = {'query_nodes': self.query_nodes,
                  'intervention_targets': self.intervention_targets,
                  'sample_queries': self.sample_queries}
        return params

    @classmethod
    def load_param_dict(cls, param_dict):
        return InterventionalDistributionsQuery(param_dict['query_nodes'], param_dict['intervention_targets'],
                                                param_dict['sample_queries'])


class Environment:
    def __init__(self, num_nodes: int,
                 mechanism_model: Optional[str] = 'gp-model',
                 frac_non_intervenable_nodes: float = None,
                 non_intervenable_nodes: Set = None,
                 num_test_samples_per_intervention: int = 50,
                 num_test_queries: int = 30,
                 interventional_queries: List[InterventionalDistributionsQuery] = None,
                 graph: nx.DiGraph = None):

        # generate unique env name
        seed = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
        self.name = self.__class__.__name__ + f'-{num_nodes}-{seed}'

        # construct graph
        self.num_nodes = num_nodes
        self.graph = self.construct_graph(num_nodes) if graph is None else graph
        self.topological_order = list(nx.topological_sort(self.graph))
        self.node_labels = sorted(list(set(self.graph.nodes)))

        # generate mechanisms
        self.mechanism_model = mechanism_model
        if mechanism_model is not None:
            mechanisms = []
            for node in self.node_labels:
                parents = get_parents(node, self.graph)
                mechanisms.append(self.create_mechanism(len(parents)))
            self.mechanisms = dict(zip(self.node_labels, mechanisms))
        else:
            self.mechanisms = None

        # optional: restrict intervenable nodes
        self.non_intervenable_nodes = set()
        if frac_non_intervenable_nodes is not None:
            num_non_intervenable_nodes = int(num_nodes * frac_non_intervenable_nodes)
            node_idc = torch.randperm(num_nodes)[:num_non_intervenable_nodes]
            self.non_intervenable_nodes = set(self.node_labels[i] for i in node_idc)
        if non_intervenable_nodes is not None:
            self.non_intervenable_nodes |= non_intervenable_nodes

        self.intervenable_nodes = set(self.node_labels) - self.non_intervenable_nodes

        # set intervention bounds for experiment design
        self.intervention_bounds = dict(zip(self.node_labels, [(-7., 7.) for _ in range(self.num_nodes)]))

        # generate observational/interventional test data
        self.observational_test_data = self.interventional_test_data = None
        self.num_test_samples_per_intervention = num_test_samples_per_intervention
        if num_test_samples_per_intervention > 0:
            self.observational_test_data = [self.sample({}, 1, num_test_samples_per_intervention)]
            self.interventional_test_data = dict()
            for node in self.node_labels:
                bounds = self.intervention_bounds[node]
                intr_values = torch.rand(num_test_samples_per_intervention) * (bounds[1] - bounds[0]) + bounds[0]
                experiments = [self.sample({node: intr_values[i]}, 1) for i in range(num_test_samples_per_intervention)]
                self.interventional_test_data.update({node: experiments})

        # generate query test data
        self.num_test_queries = num_test_queries
        self.interventional_queries = interventional_queries
        self.query_ll = torch.tensor(0.)

        if num_test_queries > 0 and self.interventional_queries is not None:
            with torch.no_grad():
                for query in self.interventional_queries:
                    experiments = []
                    for i in range(num_test_queries):
                        interventions = query.sample_intervention()
                        experiments.append(self.sample(interventions, 1))

                    query.set_sample_queries(experiments)

                query_lls = torch.zeros(num_test_queries, len(self.interventional_queries))
                for i in range(num_test_queries):
                    for query_idx, query in enumerate(self.interventional_queries):
                        query_node = query.query_nodes[0]  # ToDo: supports only single query node!!!
                        targets = query.sample_queries[i].data[query_node].squeeze(-1)
                        imll = self.interventional_mll(targets, query_node, query.sample_queries[i].interventions)
                        query_lls[i, query_idx] = imll

                self.query_ll = query_lls.sum(dim=1).mean()

    def create_mechanism(self, num_parents: int):
        if self.mechanism_model == 'gp-model':
            return GaussianProcess(num_parents, static=True) if num_parents > 0 else GaussianRootNode(static=True)

        assert False, print(f'Invalid mechanism model {self.mechanism_model}!')

    def sample(self, interventions: dict, batch_size: int, num_batches: int = 1) -> Experiment:
        data = dict()
        for node in self.topological_order:
            # check if node is intervened upon
            if node in interventions:
                samples = torch.ones(num_batches, batch_size, 1) * interventions[node]
            else:
                mech = self.mechanisms[node]

                # sample from mechanism
                parents = get_parents(node, self.graph)
                if not parents:
                    samples = mech.sample(torch.empty(num_batches, batch_size, 1))
                else:
                    x = torch.cat([data[parent] for parent in parents], dim=-1)
                    assert x.shape == (num_batches, batch_size, mech.in_size), print(f'Invalid shape {x.shape}!')
                    samples = mech.sample(x)

            # store samples
            data[node] = samples

        return Experiment(interventions, data)

    def log_likelihood(self, experiments: List[Experiment]) -> torch.Tensor:
        ll = torch.tensor(0.)
        for node in self.node_labels:
            # gather data from the experiments
            parents = get_parents(node, self.graph)
            inputs, targets = gather_data(experiments, node, parents=parents, mode='independent_samples')

            # check if we have any data for this node and compute log-likelihood
            mechanism_ll = torch.tensor(0.)
            if targets is not None:
                try:
                    mechanism_ll = self.mechanisms[node].mll(inputs, targets, prior_mode=False)
                except Exception as e:
                    print(f'Exception occured in Environment.log_likelihood() when computing LL for mechanism {node}:')
                    print(e)

            ll += mechanism_ll
        return ll

    def interventional_mll(self, targets, node: str, interventions: dict, num_mc_samples=200, reduce=True):
        assert targets.dim() == 2, print(f'Invalid shape {targets.shape}')
        num_batches, batch_size = targets.shape

        parents = get_parents(node, self.graph)
        mechanism = self.mechanisms[node]

        if len(parents) == 0:
            # if we have a root note imll is simple
            ll = mechanism.mll(None, targets, prior_mode=False, reduce=False)
            assert ll.shape == (num_batches,), print(f'Invalid shape {ll.shape}!')
            return ll.sum() if reduce else ll

        # otherwise, do MC estimate via ancestral sampling
        samples = self.sample(interventions, batch_size, num_mc_samples)
        # assemble inputs and targets
        inputs, _ = gather_data([samples], node, parents=parents, mode='independent_batches')
        inputs = inputs.unsqueeze(0).expand(num_batches, -1, -1, -1)
        assert inputs.shape == (num_batches, num_mc_samples, batch_size, len(parents))
        targets = targets.unsqueeze(1).expand(-1, num_mc_samples, batch_size)
        assert targets.shape == (num_batches, num_mc_samples, batch_size)
        # compute interventional ll
        ll = mechanism.mll(inputs, targets, prior_mode=False, reduce=False).squeeze(-1)
        assert ll.shape == (num_batches, num_mc_samples), print(f'Invalid shape: {ll.shape}')
        ll = ll.logsumexp(dim=1) - math.log(num_mc_samples)
        return ll.sum() if reduce else ll

    def construct_graph(self, num_nodes: int) -> nx.DiGraph:
        raise NotImplementedError

    def param_dict(self):
        mechanism_param_dict = {key: m.param_dict() for key, m in self.mechanisms.items()}
        if self.interventional_queries is not None:
            intr_query_param_dicts = [query.param_dict() for query in self.interventional_queries]
        else:
            intr_query_param_dicts = None
        params = {'num_nodes': self.num_nodes,
                  'mechanism_model': self.mechanism_model,
                  'graph': self.graph,
                  'name': self.name,
                  'mechanism_param_dict': mechanism_param_dict,
                  'non_intervenable_nodes': self.non_intervenable_nodes,
                  'intervention_bounds': self.intervention_bounds,
                  'num_test_samples_per_intervention': self.num_test_samples_per_intervention,
                  'observational_test_data': self.observational_test_data,
                  'interventional_test_data': self.interventional_test_data,
                  'num_test_queries': self.num_test_queries,
                  'intr_query_param_dicts': intr_query_param_dicts,
                  'query_ll': self.query_ll}
        return params

    def load_param_dict(self, param_dict):
        self.num_nodes = param_dict['num_nodes']
        self.mechanism_model = param_dict['mechanism_model']
        self.graph = param_dict['graph']
        self.topological_order = list(nx.topological_sort(self.graph))
        self.node_labels = sorted(list(set(self.graph.nodes)))
        self.name = param_dict['name']
        self.non_intervenable_nodes = param_dict['non_intervenable_nodes']
        self.intervenable_nodes = set(self.node_labels) - self.non_intervenable_nodes
        self.intervention_bounds = param_dict['intervention_bounds']
        self.num_test_samples_per_intervention = param_dict['num_test_samples_per_intervention']
        self.observational_test_data = param_dict['observational_test_data']
        self.interventional_test_data = param_dict['interventional_test_data']
        self.num_test_queries = param_dict['num_test_queries']
        self.query_ll = param_dict['query_ll']
        self.mechanisms = dict()
        for key, d in param_dict['mechanism_param_dict'].items():
            self.mechanisms[key] = self.create_mechanism(d['in_size'])
            self.mechanisms[key].load_param_dict(d)

        if param_dict['intr_query_param_dicts'] is not None:
            self.interventional_queries = []
            for query_param_dict in param_dict['intr_query_param_dicts']:
                self.interventional_queries.append(InterventionalDistributionsQuery.load_param_dict(query_param_dict))

    def save(self, path):
        torch.save(self.param_dict(), path)

    @classmethod
    def load(cls, path):
        param_dict = torch.load(path)
        env = Environment(param_dict['num_nodes'], mechanism_model=None, num_test_samples_per_intervention=0,
                          num_test_queries=0, graph=param_dict['graph'])
        env.load_param_dict(param_dict)
        return env
