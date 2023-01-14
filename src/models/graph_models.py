import functools
import math
from typing import List, Tuple, Dict, Any

import networkx as nx
import torch
import torch.distributions as dist
from torch.autograd import Function
from torch.nn.functional import logsigmoid


def generate_all_dags(num_nodes: int = 3, node_labels: List[str] = None) -> List[nx.DiGraph]:
    """Generates all directed acyclic graphs with a given number of nodes or node labels.

    Parameters
    ----------
    num_nodes : int
        The number of nodes. If `node_labels` is given this has no effect.
    node_labels : List[str]
        List of node labels. If not None, the number of nodes is inferred automatically.

    Returns
    ------
    List[nx.DiGraph]
        A list of DAGs as Networkx DiGraph objects..
    """
    # check if node labels given and create enumerative mapping
    if node_labels is not None:
        node_labels = sorted(list(set(node_labels)))
        num_nodes = len(node_labels)
        node_map = dict(zip(list(range(num_nodes)), node_labels))
    else:
        node_map = dict(zip(list(range(num_nodes)), list(range(num_nodes))))

    # check dag size feasibility
    assert num_nodes > 0, f'There is no such thing as a graph with {num_nodes} nodes.'
    assert num_nodes < 5, f'There are a lot of DAGs with {num_nodes} nodes...'

    # generate adjecency lists of all possible, simple graphs with num_nodes nodes
    adj_lists = [[]]
    for src in range(num_nodes):
        for dest in range(num_nodes):
            if src != dest:
                adj_lists = adj_lists + [[[node_map[src], node_map[dest]]] + adj_list for adj_list in adj_lists]

    # create graphs and keep only DAGs
    graphs = []
    for adj_list in adj_lists:
        graph = nx.DiGraph()
        graph.add_nodes_from(list(node_map.values()))
        graph.add_edges_from(adj_list)
        if nx.is_directed_acyclic_graph(graph):
            graphs.append(graph)

    return graphs


def get_graph_key(graph: nx.DiGraph) -> str:
    """Generates a unique string representation of a directed graph. Can be used as a dictionary key.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph for which to generate the string representation.

    Returns
    ------
    str
        A unique string representation of `graph`.
    """
    graph_str = ''
    for i, node in enumerate(sorted(graph)):
        if i > 0:
            graph_str += '|'
        graph_str += str(node) + '<-' + ','.join([str(parent) for parent in get_parents(node, graph)])

    return graph_str


def resolve_graph_key(key: str) -> nx.DiGraph:
    """Return a NetworkX DiGraph object according to the given graph key.

    Parameters
    ----------
    key : str
        The string representation of the graph to be generated.

    Returns
    ------
    nx.DiGraph
        A graph object corresponding to the given graph key.
    """
    graph = nx.DiGraph()
    mech_strings = key.split('|')
    for mstr in mech_strings:
        idx = mstr.find('<-')
        node = mstr[:idx]
        parents = mstr[idx + 2:].split(',') if len(mstr) > idx + 2 else []

        graph.add_node(node)
        for parent in parents:
            graph.add_edge(parent, node)

    return graph


def get_parents(node: str, graph: nx.DiGraph) -> List[str]:
    """Returns a list of parents for a given node in a given graph.

    Parameters
    ----------
    node : str
        The child node.
    graph : nx.DiGraph
        The graph inducing the parent set.

    Returns
    ------
    List[str]
        The list of parents.
    """
    return sorted(list(graph.predecessors(node)))


def graph_to_adj_mat(graph: nx.DiGraph, node_labels: List[str]) -> torch.Tensor:
    """Returns the adjecency matrix of the given graph as tensor.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph.
    node_labels : List[str]
        The list of node labels determining the order of the adjacency matrix.

    Returns
    ------
    torch.Tensor
        The adjacency matrix of the graph.
    """
    return torch.tensor(nx.to_numpy_array(graph, nodelist=node_labels)).float()


class CategoricalModel:
    """
    Class that represents a categorical distribution over graphs.

    Attributes
    ----------
    graphs : List[nx.DiGraph]
        List of possible DAGs for this model.
    node_labels : List[str]
        List of node labels.
    num_nodes : int
        The number of nodes in this model.
    num_graphs : int
        The number of possible DAGs for this model.
    log_probs : Dict[str, torch.Tensor]
        Dictionary of graph identifiers and corresponding log probabilities.
    """

    def __init__(self, node_labels: List[str]):
        """
        Parameters
        ----------
        node_labels : List[str]
            List of node labels.
        """
        self.node_labels = sorted(list(set(node_labels)))
        self.num_nodes = len(node_labels)
        self.graphs = generate_all_dags(node_labels=node_labels)
        self.num_graphs = len(self.graphs)
        graph_keys = [get_graph_key(graphs) for graphs in self.graphs]
        self.log_probs = dict(zip(graph_keys, -torch.log(self.num_graphs * torch.ones(self.num_graphs))))

    def log_prob(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Returns the log probability for a given graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The query graph.

        Returns
        ----------
        torch.Tensor
            The log probability.
        """
        assert get_graph_key(graph) in self.log_probs
        return self.log_probs[get_graph_key(graph)]

    def prob(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Returns the probability for a given graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The query graph.

        Returns
        ----------
        torch.Tensor
            The probability.
        """
        assert get_graph_key(graph) in self.log_probs
        return self.log_probs[get_graph_key(graph)].exp()

    def set_log_prob(self, log_prob: torch.tensor, graph: nx.DiGraph):
        """
        Sets the log probability of a given graph.

        Parameters
        ----------
        log_prob: torch.tensor
            The new log prabability.
        graph : nx.DiGraph
            The target graph.
        """
        assert log_prob.numel() == 1
        assert get_graph_key(graph) in self.log_probs

        self.log_probs[get_graph_key(graph)] = log_prob.squeeze()

    def normalize(self):
        """
        Normalizes the distribution over graphs such that the sum over all graphs probabilities equals 1.
        """
        logits = torch.stack([p for p in self.log_probs.values()])
        log_evidence = logits.logsumexp(dim=0)
        log_probs = logits - log_evidence
        self.log_probs = dict(zip(self.log_probs.keys(), log_probs))

    def entropy(self):
        """
        Returns the entropy of the categorical distribution over graphs.

        Returns
        ----------
        torch.Tensor
            The entropy.
        """
        tmp = torch.stack(list(self.log_probs.values()))
        return -(tmp.exp() * tmp).sum()

    def sample(self, num_graphs: int) -> List[nx.DiGraph]:
        """
        Samples a graph.

        Parameters
        ----------
        num_graphs: int
            Number of graphs to sample.

        Returns
        ----------
        List[nx.DiGraph]
            List of sampled graph objects.
        """
        probs = torch.stack([self.prob(graph) for graph in self.graphs])
        graph_idc = torch.multinomial(probs, num_graphs, replacement=True)
        return [self.graphs[idx] for idx in graph_idc]

    def sort_by_prob(self, descending: bool = True) -> List[nx.DiGraph]:
        """
        Returns a list of all graphs sorted by their probabilities.

        Parameters
        ----------
        descending: bool
            If true, sort in descending order, ascending otherwise.

        Returns
        ----------
        List[nx.DiGraph]
            List of sorted graph objects.
        """

        def compare(graph1, graph2):
            return (self.log_prob(graph1) - self.log_prob(graph2)).item()

        return sorted(self.graphs, key=functools.cmp_to_key(compare), reverse=descending)

    def edge_probs(self) -> torch.Tensor:
        """
        Returns the matrix of edge probabilities.

        Returns
        ----------
        torch.Tensor
            Matrix of edge probabilities.
        """
        edge_probs = torch.zeros(self.num_nodes, self.num_nodes)
        for graph in self.graphs:
            adj_mat = graph_to_adj_mat(graph, self.node_labels)
            edge_probs += self.prob(graph) * adj_mat

        return edge_probs

    def get_mc_graphs(self, mode: str, num_mc_graphs: int = 20) -> Tuple[List[nx.DiGraph], torch.Tensor]:
        """
        Returns a set of graphs and corresponding log-weights for Monte Carlo estimation.

        Parameters
        ----------
        mode: str
            There are three strategies for generating the MC graph set:
              'full': Returns all graphs and their log-probabilities as log-weights.
              'sampling': Returns `num_mc_graphs` samples from the distribution over graphs with uniform weights.
              'n-best': Returns the `num_mc_graphs` graphs with the highest probabilities weighted according to their
                        re-normalized probabilities.
        num_mc_graphs: int
            The size of the returned set of graphs (see description above). Has no effect when mode is 'full'.

        Returns
        ----------
        List[nx.DiGraph], torch.Tensor
            The set of MC graphs and their corresponding log-weights.
        """
        if mode not in {'full', 'sampling', 'n-best'}:
            print('Invalid sampling mode >' + mode + '<. Doing <full> instead.')
            mode = 'full'

        if mode == 'sampling':
            graphs = self.sample(num_mc_graphs)
            log_weights = -torch.ones(num_mc_graphs) * math.log(num_mc_graphs)
        elif mode == 'n-best':
            graphs = self.sort_by_prob()[:num_mc_graphs]
            log_weights = [self.log_prob(graph) for graph in graphs]
            log_weights = torch.log_softmax(torch.stack(log_weights), dim=0)
        else:
            graphs = self.graphs
            log_weights = [self.log_prob(graph) for graph in graphs]
            log_weights = torch.stack(log_weights)

        return graphs, log_weights

    def param_dict(self) -> Dict[str, Any]:
        """
        Returns the current parameters of the an instance of this class as a dictionary.

        Returns
        ----------
        Dict[str, Any]
            Parameter dictionary.
        """
        params = {'node_labels': self.node_labels,
                  'num_nodes': self.num_nodes,
                  'graphs': self.graphs,
                  'num_graphs': self.num_graphs,
                  'log_probs': self.log_probs}
        return params

    def load_param_dict(self, param_dict: Dict[str, Any]):
        """
        Sets the parameters of this class instance with the parameter values given in `param_dict`.

        Parameters
        ----------
        param_dict : Dict[str, Any]
            Parameter dictionary.
        """
        self.node_labels = param_dict['node_labels']
        self.num_nodes = param_dict['num_nodes']
        self.graphs = param_dict['graphs']
        self.num_graphs = param_dict['num_graphs']
        self.log_probs = param_dict['log_probs']


class DiBSModel:
    def __init__(self, node_labels: List[str], embedding_size: int, num_particles: int, std: float = 1.):
        self.node_labels = sorted(list(set(node_labels)))
        self.num_nodes = len(self.node_labels)
        self.node_id_to_label_dict = dict(zip(list(range(self.num_nodes)), node_labels))
        self.node_label_to_id_dict = dict(zip(node_labels, list(range(self.num_nodes))))
        self.embedding_size = embedding_size
        self.num_particles = num_particles
        self.normal_prior_std = std
        self.particles = self.sample_initial_particles(num_particles)

    def _check_particle_shape(self, z: torch.Tensor):
        assert z.dim() == 4 and z.shape[1:] == (self.embedding_size, self.num_nodes, 2), print(z.shape)

    def sample_initial_particles(self, num_particles) -> torch.Tensor:
        # sample particles from normal prior
        normal = torch.distributions.Normal(0., self.normal_prior_std)
        particles = normal.sample((num_particles, self.embedding_size, self.num_nodes, 2))
        particles.requires_grad_(True)
        self._check_particle_shape(particles)
        return particles

    def edge_logits(self, alpha: float = 1.):
        return alpha * torch.einsum('ikj,ikl->ijl', self.particles[..., 0], self.particles[..., 1])

    def edge_probs(self, alpha: float = 1.):
        # compute edge probs
        edge_probs = torch.sigmoid(self.edge_logits(alpha))

        # set probs of self loops to 0
        mask = torch.eye(self.num_nodes).repeat(self.num_particles, 1, 1).bool()
        edge_probs[mask] = 0.
        return edge_probs

    def edge_log_probs(self, alpha: float = 1.):
        return logsigmoid(self.edge_logits(alpha))

    def log_generative_prob(self, adj_mats: torch.Tensor, alpha: float = 1., batch_mode=True):
        assert adj_mats.dim() == 4 and adj_mats.shape[2:] == (self.num_nodes, self.num_nodes)
        assert adj_mats.shape[0] == self.particles.shape[0] or not batch_mode
        logits = self.edge_logits(alpha)
        log_edge_probs = logsigmoid(logits)
        log_not_edge_probs = log_edge_probs - logits  # =logsigmoid(-logits) = log(1-sigmoid(logits))

        # set probs of self loops to 0
        mask = torch.eye(self.num_nodes).repeat(self.num_particles, 1, 1).bool()
        log_edge_probs = torch.where(mask, torch.tensor(0.), log_edge_probs)
        log_not_edge_probs = torch.where(mask, torch.tensor(0.), log_not_edge_probs)

        if batch_mode:
            graph_log_probs = torch.einsum('hijk,hjk->hi', adj_mats, log_edge_probs) + \
                              torch.einsum('hijk,hjk->hi', (1. - adj_mats), log_not_edge_probs)
        else:
            graph_log_probs = torch.einsum('hijk,ljk->lhi', adj_mats, log_edge_probs) + \
                              torch.einsum('hijk,ljk->lhi', (1. - adj_mats), log_not_edge_probs)

        return graph_log_probs

    def unnormalized_log_prior(self, alpha: float = 1., beta: float = 1.) -> torch.Tensor:
        normal = torch.distributions.Normal(0., self.normal_prior_std)

        ec = self.expected_cyclicity(alpha)
        log_prior = normal.log_prob(self.particles).sum(dim=(1, 2, 3)) / self.particles[0].numel() - beta * ec
        return log_prior.float()

    def expected_cyclicity(self, alpha: float = 1., num_samples: int = 100) -> torch.Tensor:
        adj_mats = self.sample_soft_graphs(num_samples, alpha)
        scores = AcyclicityScore.apply(adj_mats)
        return scores.sum(dim=-1) / num_samples

    def sample_soft_graphs(self, num_samples: int, alpha: float = 1.):
        edge_logits = self.edge_logits(alpha)

        transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=0., scale=1.)]
        logistic = torch.distributions.TransformedDistribution(dist.Uniform(0, 1), transforms)
        reparam_logits = logistic.rsample((num_samples, *edge_logits.shape)) + \
                         edge_logits.unsqueeze(0).expand(num_samples, -1, -1, -1)
        soft_adj_mats = torch.sigmoid(reparam_logits).permute(1, 0, 2, 3)

        # eliminate self loops
        mask = torch.eye(self.num_nodes).repeat(self.num_particles, num_samples, 1, 1).bool()
        soft_adj_mats = torch.where(mask, torch.tensor(0.), soft_adj_mats)
        return soft_adj_mats

    def sample_graphs(self, num_samples: int, alpha: float = 1., fixed_edges: List[Tuple[int, int]] = None) \
        -> Tuple[List[List[nx.DiGraph]], torch.Tensor]:
        # compute bernoulli probs from latent particles
        edge_probs = self.edge_probs(alpha)

        # modify probs
        if fixed_edges is not None:
            for i, j in fixed_edges:
                edge_probs[:, i, j] = 1.

        # sample adjacency matrices and generate graph objects
        adj_mats = torch.bernoulli(edge_probs.unsqueeze(1).expand(-1, num_samples, -1, -1))
        graphs = [[self.adj_mat_to_graph(adj_mats[pidx, sidx]) for sidx in range(num_samples)] for pidx in range(
            self.num_particles)]
        return graphs, adj_mats

    def dagify_graphs(self, graphs: List[List[nx.DiGraph]], adj_mats: torch.Tensor):
        """Uses a simple heuristic to 'dagify' cyclic graphs in-place. Note: this can be handy for testing and
        debugging during developement, but should not be necessary when the DiBS model is trained properly (as it
        should then almost always return DAGs when sampling).

        Parameters
        ----------
        graphs : List[List[nx.DiGraph]]
            Nested lists of graph objects.
        adj_mats : torch.Tensor
            Tensor of adjacency matrices corresponding to the graph objects in `graphs`.
        """
        edge_probs = self.edge_probs()
        for particle_idx in range(self.num_particles):
            num_dagified = 0
            for graph_idx, graph in enumerate(graphs[particle_idx]):
                # check if the graph is cyclic
                if not nx.is_directed_acyclic_graph(graphs[particle_idx][graph_idx]):
                    edges, _ = self.sort_edges(adj_mats[particle_idx, graph_idx], edge_probs[particle_idx])

                    graph = nx.DiGraph()
                    graph.add_nodes_from(self.node_labels)
                    adj_mats[particle_idx, graph_idx] = torch.zeros(self.num_nodes, self.num_nodes)
                    for edge_idx, edge in enumerate(edges):
                        source_node = self.node_id_to_label_dict[edge[0]]
                        sink_node = self.node_id_to_label_dict[edge[1]]
                        if not nx.has_path(graph, sink_node, source_node):
                            # if there is no path from the target to the source node, we can safely add the edge to
                            # the graph without creating a cycle
                            graph.add_edge(source_node, sink_node)
                            adj_mats[particle_idx, graph_idx, edge[0], edge[1]] = 1

                    graphs[particle_idx][graph_idx] = graph
                    num_dagified += 1

            if num_dagified > 0:
                print(f'Dagified {num_dagified} graphs of the {particle_idx + 1}-th particle!')

    def sort_edges(self, adj_mat: torch.Tensor, edge_weights: torch.Tensor, descending=True):
        edges = [(i, j) for i in range(self.num_nodes) for j in range(self.num_nodes) if adj_mat.bool()[i, j]]
        weights = edge_weights[adj_mat.bool()]
        weights, idc = torch.sort(weights, descending=descending)
        edges = [edges[idx] for idx in idc]
        return edges, weights

    def adj_mat_to_graph_key(self, adj_mat: torch.Tensor):
        assert adj_mat.shape == (self.num_nodes, self.num_nodes)
        key = '|'.join(
            [self.node_labels[i] + '<-' + ','.join(
                [self.node_labels[j] for j in range(self.num_nodes) if adj_mat[j, i] > 0.5]) for i in
             range(self.num_nodes)])
        return key

    def adj_mat_to_graph(self, adj_mat: torch.Tensor) -> nx.DiGraph:
        assert adj_mat.shape == (self.num_nodes, self.num_nodes)
        graph = nx.from_numpy_array(adj_mat.int().numpy(), create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, self.node_id_to_label_dict)
        return graph

    def graph_to_adj_mat(self, graph: nx.DiGraph) -> torch.Tensor:
        return graph_to_adj_mat(graph, self.node_labels)

    def get_limit_graphs(self):
        # round edge probs to get limit graphs
        edge_probs = self.edge_probs()
        adj_mats = edge_probs.round().unsqueeze(1)
        graphs = [[self.adj_mat_to_graph(adj_mats[i, 0])] for i in range(self.num_particles)]
        return graphs, adj_mats

    def particle_similarities(self, bandwidth=1.):
        distances = [(self.particles - self.particles[i:i + 1].detach()) ** 2 for i in range(self.num_particles)]
        similarities = [(-d.sum(dim=(1, 2, 3)) / bandwidth).exp() for d in distances]
        kernel_mat = torch.stack(similarities, dim=1)
        return kernel_mat

    def param_dict(self):
        params = {'node_id_to_label_dict': self.node_id_to_label_dict,
                  'node_label_to_id_dict': self.node_label_to_id_dict,
                  'embedding_size': self.num_particles,
                  'normal_prior_std': self.normal_prior_std,
                  'particles': self.particles}
        return params

    def load_param_dict(self, param_dict):
        self.node_id_to_label_dict = param_dict['node_id_to_label_dict']
        self.node_label_to_id_dict = param_dict['node_label_to_id_dict']
        self.node_labels = sorted(list(self.node_label_to_id_dict.keys()))
        self.num_nodes = len(self.node_labels)
        self.particles = param_dict['particles']
        self.num_particles = self.particles.shape[0]
        self.embedding_size = self.particles.shape[1]
        self.normal_prior_std = param_dict['normal_prior_std']


class AcyclicityScore(Function):
    @staticmethod
    def forward(ctx, adj_mat: torch.Tensor, round_edge_weights=False):
        assert adj_mat.dim() >= 3 and adj_mat.shape[-1] == adj_mat.shape[-2], print(
            f'Ill-shaped input: {adj_mat.shape}')
        num_nodes = adj_mat.shape[-1]
        eyes = torch.eye(num_nodes).double().expand_as(adj_mat)
        tmp = eyes + (adj_mat.round().double() if round_edge_weights else adj_mat) / num_nodes

        tmp_pow = tmp.matrix_power(num_nodes - 1)
        ctx.grad = tmp_pow.transpose(-1, -2)
        score = (tmp_pow @ tmp).diagonal(dim1=-2, dim2=-1).sum(dim=-1) - num_nodes
        return score

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        return ctx.grad * grad_output[..., None, None], None
