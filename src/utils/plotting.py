import math
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import scipy.stats as sst
import torch


def init_plot_style():
    """Initialize the plot style for pyplot.
    """
    plt.rcParams.update({'figure.figsize': (12, 9)})
    plt.rcParams.update({'lines.linewidth': 2})
    plt.rcParams.update({'lines.markersize': 25})
    plt.rcParams.update({'lines.markeredgewidth': 2})
    plt.rcParams.update({'axes.labelpad': 10})
    plt.rcParams.update({'xtick.major.width': 2.5})
    plt.rcParams.update({'xtick.major.size': 15})
    plt.rcParams.update({'xtick.minor.size': 10})
    plt.rcParams.update({'ytick.major.width': 2.5})
    plt.rcParams.update({'ytick.minor.width': 2.5})
    plt.rcParams.update({'ytick.major.size': 15})
    plt.rcParams.update({'ytick.minor.size': 15})

    # for font settings see also https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
    plt.rcParams.update({'font.size': 50})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams['text.latex.preamble'] = '\n'.join([
        r'\usepackage{amsmath,amssymb,amsfonts,amsthm}',
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',  # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ])


def parse_file_name(filename: str):
    tokens = filename.split('-')
    if tokens[-2] == 'exp':
        job_id = tokens[-3]
        env_id = tokens[-4]
        exp_num = int(tokens[-1][:-4])
    else:
        job_id = tokens[-1][:-4]
        env_id = tokens[-2]
        exp_num = 0
    return env_id, job_id, exp_num


class Simulation:
    def __init__(self, env_name: str, num_nodes: int, timestamp: str, abci_model: str, policy: str,
                 num_experiments: int,
                 plot_kwargs: Optional[dict] = None):
        self.env_name = env_name
        self.num_nodes = num_nodes
        self.timestamp = timestamp
        self.abci_model = abci_model
        self.policy = policy
        self.num_experiments = num_experiments
        self.stats = None
        self.plot_kwargs = dict() if plot_kwargs is None else plot_kwargs

    def get_result_files(self, base_dir: str = '../results/'):
        results_dir = base_dir + f'{self.env_name}/{self.num_nodes}_nodes/' \
                                 f'{self.timestamp}_{self.abci_model}_{self.policy}/'
        abci_files = [entry for entry in os.scandir(results_dir) if
                      entry.is_file() and os.path.basename(entry)[-4:] == '.pth']

        common_env_files = dict()
        for f in abci_files:
            env_id, job_id, exp_num = parse_file_name(os.path.basename(f))
            if exp_num != self.num_experiments:
                continue

            if env_id in common_env_files:
                common_env_files[env_id].append(os.path.abspath(f))
            else:
                common_env_files[env_id] = [os.path.abspath(f)]

        return common_env_files

    def load_results(self, stats_names: List[str], base_dir: str = '../results/'):
        common_env_files = self.get_result_files(base_dir)
        num_environments = len(common_env_files)
        print(f'Loading results from {num_environments} environments.')

        stats_lists = dict()
        for env_id, env_files in common_env_files.items():
            print(f'Got {len(env_files)} runs for environment {env_id}.')
            env_wise_stat_lists = dict()
            for abci_file in env_files:
                param_dict = torch.load(abci_file)

                for stat_name in stats_names:
                    if stat_name in {'interventional_test_ll', 'interventional_kld'}:
                        stat_token = stat_name + '_lists'
                    else:
                        stat_token = stat_name + '_list'

                    if stat_token not in param_dict:
                        print(f'No results for {stat_token} available in {abci_file}!')
                        continue

                    if stat_name == 'interventional_test_ll':
                        data = -torch.tensor(list(param_dict[stat_token].values())).sum(dim=0)
                    elif stat_name == 'interventional_kld':
                        data = torch.tensor(list(param_dict[stat_token].values())).mean(dim=0)
                    else:
                        data = torch.tensor(param_dict[stat_token])
                        if stat_name in {'graph_ll', 'observational_test_ll'}:
                            data = -data

                    if stat_name in env_wise_stat_lists:
                        env_wise_stat_lists[stat_name].append(data)
                    else:
                        env_wise_stat_lists[stat_name] = [data]

            # aggregate env-wise results
            for stat_name in env_wise_stat_lists:
                with torch.no_grad():
                    data = torch.stack(env_wise_stat_lists[stat_name], dim=0)

                reduce = lambda x: x
                # reduce = lambda x: x.mean(dim=0, keepdims=True)

                if stat_name in stats_lists:
                    stats_lists[stat_name].append(reduce(data))
                else:
                    stats_lists[stat_name] = [reduce(data)]

        self.stats = {stat_name: torch.cat(stat_list, dim=0) for stat_name, stat_list in stats_lists.items()}
        return self.stats

    def plot_simulation_data(self, ax, stat_name: str):
        if self.stats is None:
            print('Nothing to plot...')
            return

        data = self.stats[stat_name]
        num_envs, num_exps = data.shape
        exp_numbers = torch.arange(1, num_exps + 1)

        # compute 95% CIs
        mean = data.mean(dim=0)
        std_err = data.std(unbiased=True, dim=0) / math.sqrt(num_envs) + 1e-8
        lower, upper = sst.t.interval(.95, df=num_envs - 1, loc=mean, scale=std_err)

        ax.plot(exp_numbers, mean.detach(), **self.plot_kwargs)
        # if self.policy not in {'observational', 'random-fixed-value'}:
        ax.fill_between(exp_numbers, upper, lower, alpha=0.2, color=self.plot_kwargs['c'])
