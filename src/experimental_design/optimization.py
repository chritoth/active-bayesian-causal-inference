import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import ExactMarginalLogLikelihood


def gp_ucb(utility: callable, bounds: torch.Tensor, num_total_candidates=8, num_initial_candidates=1):
    assert bounds.shape == (2, 1), print(bounds.shape)

    # generate initial candidates
    candidate_list = [(torch.rand(1) * (bounds[1] - bounds[0]) + bounds[0]).unsqueeze(1) for _ in
                      range(num_initial_candidates)]
    utility_list = [utility(candidate.squeeze().item()).squeeze() for candidate in candidate_list]

    # search for better candidates
    try:
        xrange = torch.linspace(bounds[0].item(), bounds[1].item(), 50).view(-1, 1, 1)
        for i in range(num_total_candidates - num_initial_candidates):
            # fit acquisition function GP
            gp = SingleTaskGP(torch.cat(candidate_list).view(-1, 1), torch.stack(utility_list).view(-1, 1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll.train()
            fit_gpytorch_torch(mll, options={'disp': False, 'maxiter': 50})
            # fit_gpytorch_scipy(mll)

            # get new candidate
            ucb = UpperConfidenceBound(gp, beta=1.)(xrange)
            shuffle_idc = torch.randperm(ucb.numel())
            argmax = ucb[shuffle_idc].argmax()
            candidate = xrange[shuffle_idc[argmax]]
            # candidate, acq_value = optimize_acqf(ucb, bounds=bounds, q=1, num_restarts=1, raw_samples=100)

            # record candidate and objective
            candidate_list.append(candidate)
            utility_list.append(utility(candidate.squeeze().item()).squeeze())
    except Exception as e:
        print('Exception occured when running GP UCB:')
        print(e)
        print('Continuing with the best candidate from ', candidate_list)
        print('with utilities ', utility_list)

    # return best candidate and objective values
    best_candidate = candidate_list[torch.stack(utility_list).argmax().item()].squeeze()
    best_utility = torch.stack(utility_list).max().item()
    return best_candidate, best_utility


def random_search(utility: callable, bounds: torch.Tensor, num_candidates=10):
    assert bounds.shape == (2, 1), print(bounds.shape)

    # generate initial candidates
    candidate_list = [torch.rand(1) * (bounds[1] - bounds[0]) + bounds[0] for _ in range(num_candidates)]
    utility_list = [utility(candidate.item()).squeeze() for candidate in candidate_list]

    # return best candidate and objective values
    best_candidate = candidate_list[torch.stack(utility_list).argmax().item()]
    best_utility = torch.stack(utility_list).max().item()
    return best_candidate, best_utility


def grid_search(utility: callable, bounds: torch.Tensor, num_candidates=10):
    assert bounds.shape == (2, 1), print(bounds.shape)

    # generate initial candidates
    candidate_list = torch.linspace(bounds[0].item(), bounds[1].item(), num_candidates)
    utility_list = [utility(candidate.item()).squeeze() for candidate in candidate_list]

    # return best candidate and objective values
    best_candidate = candidate_list[torch.stack(utility_list).argmax().item()]
    best_utility = torch.stack(utility_list).max().item()
    return best_candidate, best_utility
