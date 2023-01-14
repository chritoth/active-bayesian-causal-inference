from typing import Dict, List

import networkx as nx

from src.models.graph_models import get_parents
from src.models.mechanisms import *


class Experiment:
    def __init__(self, interventions: dict, data: Dict[str, torch.tensor]):
        num_batches, batch_size = list(data.values())[0].shape[0:2]
        assert all([node_data.shape == (num_batches, batch_size, 1) for node_data in data.values()])
        self.interventions = interventions
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data = data


def gather_data(experiments: List[Experiment], node: str, graph: nx.DiGraph = None, parents: List[str] = None,
                mode: str = 'joint'):
    assert graph is not None or parents is not None
    assert mode in {'joint', 'independent_batches', 'independent_samples'}, print('Invalid gather mode: ', mode)

    # gather targets
    if mode == 'independent_batches':
        batch_size = experiments[0].batch_size
        assert all([experiment.batch_size == batch_size for experiment in experiments]), print('Batch size mismatch!')
        targets = [exp.data[node].squeeze(-1) for exp in experiments if node not in exp.interventions]
    elif mode == 'independent_samples':
        targets = [exp.data[node].view(-1, 1) for exp in experiments if node not in exp.interventions]
    else:  # mode == 'joint'
        targets = [exp.data[node].reshape(-1) for exp in experiments if node not in exp.interventions]

    # check if we have data for this node
    if not targets:
        return None, None
    targets = torch.cat(targets, dim=0)

    # check if we have parents
    parents = sorted(parents) if graph is None else get_parents(node, graph)
    if not parents:
        return None, targets

    # gather parent data
    num_parents = len(parents)
    if mode == 'independent_batches':
        inputs = [torch.cat([experiment.data[parent] for parent in parents], dim=-1) for experiment in experiments if
                  node not in experiment.interventions]
    elif mode == 'independent_samples':
        inputs = [torch.cat([experiment.data[parent] for parent in parents], dim=-1).view(-1, 1, num_parents) for
                  experiment in experiments if node not in experiment.interventions]
    else:  # mode == 'joint'
        inputs = [torch.cat([experiment.data[parent] for parent in parents], dim=-1).view(-1, num_parents) for
                  experiment in experiments if node not in experiment.interventions]

    inputs = torch.cat(inputs, dim=0)

    return inputs, targets
