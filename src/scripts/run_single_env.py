import argparse
import os
import random
import socketserver
import string
import time

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from src.abci_categorical_gp import ABCICategoricalGP
from src.abci_dibs_gp import ABCIDiBSGP
from src.environments.generic_environments import *

MODELS = {'abci-dibs-gp', 'abci-dibs-gp-linear', 'abci-categorical-gp', 'abci-categorical-gp-linear'}


def spawn_abci_model(abci_model, env, policy, num_workers):
    assert abci_model in MODELS, print(f'Invalid ABCI model {abci_model}')

    if abci_model == 'abci-dibs-gp':
        return ABCIDiBSGP(env, policy, num_workers=num_workers)

    if abci_model == 'abci-dibs-gp-linear':
        return ABCIDiBSGP(env, policy, num_workers=num_workers, linear=True)

    if abci_model == 'abci-categorical-gp':
        return ABCICategoricalGP(env, policy, num_workers=num_workers)

    if abci_model == 'abci-categorical-gp-linear':
        return ABCICategoricalGP(env, policy, num_workers=num_workers, linear=True)


def get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
    return free_port


def generate_job_id():
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(6)])


def run_worker(rank: int, env: Environment, master_port: str, output_dir: str, policy: str, num_experiments: int,
               batch_size: int, num_initial_obs_samples: int, num_workers: int, job_id: str, abci_model: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

    if rank == 0:
        rpc.init_rpc('Experimenter',
                     rank=rank,
                     world_size=num_workers,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=num_workers, rpc_timeout=0))
        try:
            abci = spawn_abci_model(abci_model, env, policy, num_workers)
            abci.run(num_experiments, batch_size, num_initial_obs_samples=num_initial_obs_samples, outdir=output_dir,
                     job_id=job_id)
        except Exception as e:
            print(e)
        else:
            outpath = f'{output_dir}{abci_model}-{policy}-{env.name}-{job_id}-exp-{num_experiments}.pth'
            abci.save(outpath)
    else:
        rpc.init_rpc(f'ExperimentDesigner{rank}',
                     rank=rank,
                     world_size=num_workers,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=num_workers, rpc_timeout=0))

    rpc.shutdown()


def run_single_env(env_file: str, output_dir: str, policy: str, num_experiments: int, batch_size: int,
                   num_initial_obs_samples: int, num_workers: int, abci_model: str):
    print(torch.__config__.parallel_info())
    mp.set_sharing_strategy('file_system')

    # load env
    env = Environment.load(env_file)

    assert abci_model in MODELS, print(f'Invalid ABCI model {abci_model}!')
    job_id = generate_job_id()

    # run benchmark env
    print('\n-------------------------------------------------------------------------------------------')
    print(f'--------- Running {abci_model.upper()} ({policy}) on Environment {env.name} w/ job ID {job_id} ----------')
    print(f'--------- Number of Experiments: {num_experiments} ----------')
    print(f'--------- Batch Size: {batch_size} ----------')
    print(f'--------- Number of Initial Observational Samples: {num_initial_obs_samples} ----------')
    print(f'--------- Starting time: {time.strftime("%H:%M:%S")} ----------')
    print('-------------------------------------------------------------------------------------------\n')

    if num_workers > 1:
        master_port = str(get_free_port())
        print(f'Starting {num_workers} workers on port ' + master_port)
        try:
            mp.spawn(
                run_worker,
                args=(env, master_port, output_dir, policy, num_experiments, batch_size, num_initial_obs_samples,
                      num_workers, job_id, abci_model),
                nprocs=num_workers,
                join=True
            )
        except Exception as e:
            print(e)
    else:
        try:
            abci = spawn_abci_model(abci_model, env, policy, num_workers=1)
            abci.run(num_experiments, batch_size, num_initial_obs_samples=num_initial_obs_samples, outdir=output_dir,
                     job_id=job_id)
        except Exception as e:
            print(e)
        else:
            outpath = f'{output_dir}{abci_model}-{policy}-{env.name}-{job_id}-exp-{num_experiments}.pth'
            abci.save(outpath)


# parse arguments when run from shell
if __name__ == "__main__":
    parser = argparse.ArgumentParser('ABCI usage on single environment:')
    parser.add_argument('env_file', type=str, help=f'Path to environment file.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('policy', type=str, help=f'ABCI policy.')
    parser.add_argument('--num_experiments', type=int, default=50, help='Number of experiments per environment.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples drawn in each experiment.')
    parser.add_argument('--num_initial_obs_samples', type=int, default=0,
                        help='Number of initial observational samples drawn before policy becomes active.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker threads per environment.')
    parser.add_argument('--model', default='abci-dibs-gp', type=str, choices=MODELS, help=f'Available models: {MODELS}')

    args = vars(parser.parse_args())
    run_single_env(args['env_file'], args['output_dir'], args['policy'], args['num_experiments'], args['batch_size'],
                   args['num_initial_obs_samples'], args['num_workers'], args['model'])
