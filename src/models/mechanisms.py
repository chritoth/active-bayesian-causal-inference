import math

import gpytorch
import torch
import torch.distributions as dist
from torch.nn import Module


class Mechanism(Module):
    """
    Class that represents a generic mechanism including a likelihood/noise model in an SCM.

    Attributes
    ----------
    in_size : int
            Number of mechanism inputs.
    """

    def __init__(self, in_size: int):
        """
        Parameters
        ----------
        in_size : int
            Number of mechanism inputs.
        """
        super().__init__()
        self.in_size = in_size

    def _check_args(self, inputs: torch.Tensor = None, targets: torch.Tensor = None):
        """
        Checks the generic argument shapes (inputs and targets) and their compatibility.

        Parameters
        ----------
        inputs : torch.Tensor
            Mechanism inputs.
        targets : torch.Tensor
            Mechanism targets, e.g., for evaluating the marginal log-likelihood.
        """
        if inputs is not None:
            assert inputs.dim() >= 2 and inputs.shape[-1] == self.in_size, print(
                f'Ill-shaped inputs: {inputs.shape}')
        if targets is not None:
            assert targets.dim() >= 1, print(f'Ill-shaped targets: {targets.shape}')
        if targets is not None and inputs is not None:
            assert inputs.shape[:-1] == targets.shape, print(f'Batch size mismatch: {inputs.shape} vs.'
                                                             f' {targets.shape}')

    def forward(self, inputs: torch.Tensor, prior_mode: bool = False):
        """
        Computes the mechanism output for a given input tensor. Must be implemented by all child classes.

        Parameters
        ----------
        inputs : torch.Tensor
            Mechanism inputs.
        prior_mode : bool
            Whether to evaluate the mechanism with prior or posterior parameters.
        """
        raise NotImplementedError

    def sample(self, inputs: torch.Tensor, prior_mode: bool = False):
        """
        Generates samples a given input tensor according to the implemented likelihood model. Must be implemented by
        all child classes.

        Parameters
        ----------
        inputs : torch.Tensor
            Mechanism inputs.
        prior_mode : bool
            Whether to evaluate the mechanism with prior or posterior parameters.
        """
        raise NotImplementedError


class GaussianRootNode(Mechanism):
    def __init__(self, mu_0: float = 0., kappa_0: float = 0.1, alpha_0: float = 50., beta_0: float = 25.,
                 static=False):
        super().__init__(in_size=0)

        # init prior and posterior hyper-parameters
        self.mu_0 = self.mu_n = torch.tensor(mu_0)
        self.kappa_0 = self.kappa_n = torch.tensor(kappa_0)
        self.alpha_0 = self.alpha_n = torch.tensor(alpha_0)
        self.beta_0 = self.beta_n = torch.tensor(beta_0)
        self.lam_0 = None
        self.train_targets = None

        self.static = static
        if static:
            self.init_as_static()

    def compute_posterior_params(self, targets: torch.Tensor, prior_mode=False):
        self._check_args(targets=targets)

        full_targets = targets
        if not prior_mode and self.train_targets is not None:
            full_targets = torch.cat((targets, self.train_targets.expand(*targets.shape[:-1], -1)), dim=-1)

        n = full_targets.shape[-1]
        empirical_means = full_targets.mean(dim=-1)

        kappa_n = self.kappa_0 + n
        mu_n = (self.kappa_0 * self.mu_0 + n * empirical_means) / kappa_n
        alpha_n = self.alpha_0 + 0.5 * n
        beta_n = self.beta_0 + 0.5 * (full_targets - empirical_means.unsqueeze(-1)).pow_(2).sum(dim=-1) + \
                 0.5 * self.kappa_0 * n * (empirical_means - self.mu_0).pow_(2) / kappa_n

        return mu_n, kappa_n.expand(mu_n.shape), alpha_n.expand(mu_n.shape), beta_n

    def init_as_static(self):
        self.lam_0 = dist.Gamma(self.alpha_0, self.beta_0).sample()
        self.mu_0 = dist.Normal(0., (self.kappa_0 * self.lam_0).pow(-0.5)).sample()

    def set_data(self, inputs: torch.Tensor, targets: torch.Tensor):
        self._check_args(targets=targets)
        assert targets.dim() == 1, print('Can only work with one set of posterior params!')
        self.train_targets = targets
        self.mu_n, self.kappa_n, self.alpha_n, self.beta_n = self.compute_posterior_params(targets, prior_mode=True)

    def forward(self, inputs: torch.Tensor, prior_mode=False):
        assert inputs.dim() >= 2
        output_shape = (*inputs.shape[:-1], 1)

        if self.static or prior_mode:
            return self.mu_0 * torch.ones(output_shape)

        return self.mu_n * torch.ones(output_shape)

    def sample(self, inputs: torch.Tensor, prior_mode=False):
        assert inputs.dim() >= 2
        output_shape = (*inputs.shape[:-1], 1)

        if self.static:
            # sample from true distribution
            y_dist = dist.Normal(self.mu_0, self.lam_0.pow(-0.5))
            return y_dist.sample(torch.Size(output_shape))

        # sample from marginal likelihood
        if prior_mode:
            mu_n, kappa_n, alpha_n, beta_n = (self.mu_0, self.kappa_0, self.alpha_0, self.beta_0)
        else:
            mu_n, kappa_n, alpha_n, beta_n = (self.mu_n, self.kappa_n, self.alpha_n, self.beta_n)

        lambdas = dist.Gamma(alpha_n, beta_n).sample(output_shape[:-2])
        mus = dist.Normal(mu_n.expand_as(lambdas), (kappa_n * lambdas).pow(-0.5)).sample()

        y_dist = dist.Normal(mus, lambdas.pow(-0.5))
        samples = y_dist.sample(output_shape[-2:-1]).unsqueeze(-1).transpose(0, -1).view(output_shape)
        return samples

    def mll(self, inputs: torch.Tensor, targets: torch.Tensor, prior_mode=False, reduce=True):
        self._check_args(targets=targets)
        output_shape = targets.shape[:-1]
        if self.static:
            # evaluate true log-likelihood
            y_dist = dist.Normal(self.mu_0, self.lam_0.pow(-0.5))
            lls = y_dist.log_prob(targets).squeeze(-1)
        else:
            if prior_mode:
                kappa_n, alpha_n, beta_n = (self.kappa_0, self.alpha_0, self.beta_0)
            else:
                kappa_n, alpha_n, beta_n = (self.kappa_n, self.alpha_n, self.beta_n)

            _, kappa_m, alpha_m, beta_m = self.compute_posterior_params(targets, prior_mode)
            lls = torch.lgamma(alpha_m) - torch.lgamma(alpha_n) + alpha_n * beta_n.log() - alpha_m * beta_m.log() + \
                  0.5 * (kappa_n.log() - kappa_m.log()) - 0.5 * targets.shape[-1] * math.log(2. * math.pi)

        assert lls.shape == output_shape, print(lls.shape)
        if reduce:
            return lls.sum()
        return lls

    def expected_noise_entropy(self, prior_mode: bool = False) -> torch.Tensor:
        if self.static:
            return dist.Normal(self.mu_0, self.lam_0.pow(-0.5)).entropy()

        # expected noise entropy exact
        alpha, beta = (self.alpha_0, self.beta_0) if prior_mode else (self.alpha_n, self.beta_n)
        return 0.5 * (math.log(2. * math.pi * math.e) - torch.digamma(alpha) + beta.log())

        # expected noise entropy point estimate (mean variance of inverse gamma posterior)
        # return 0.5 * (2. * math.pi * beta/(alpha + 1.) * math.e).log().squeeze()

    def param_dict(self):
        params = {'in_size': 0,
                  'mu_0': self.mu_0,
                  'kappa_0': self.kappa_0,
                  'alpha_0': self.alpha_0,
                  'beta_0': self.beta_0,
                  'lam_0': self.lam_0,
                  'mu_n': self.mu_n,
                  'kappa_n': self.kappa_n,
                  'alpha_n': self.alpha_n,
                  'beta_n': self.beta_n,
                  'static': self.static}

        return params

    def load_param_dict(self, param_dict):
        self.mu_0 = param_dict['mu_0']
        self.kappa_0 = param_dict['kappa_0']
        self.alpha_0 = param_dict['alpha_0']
        self.beta_0 = param_dict['beta_0']
        self.lam_0 = param_dict['lam_0']
        self.mu_n = param_dict['mu_n']
        self.kappa_n = param_dict['kappa_n']
        self.alpha_n = param_dict['alpha_n']
        self.beta_n = param_dict['beta_n']
        self.static = param_dict['static']


class GaussianProcess(Mechanism):
    class ExactGPModelRQKernel(gpytorch.models.ExactGP):
        # ATTENTION: do not name the HP priors "noise_prior", "outputscale_prior" or "lengthscale_prior"
        noise_var_prior = dist.Gamma(50., 500.)
        outscale_prior = dist.Gamma(100., 10.)

        def __init__(self, train_x, train_y, likelihood, in_size: int):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

            # draw kernel parameters
            self.lscale_prior = dist.Gamma(30. * in_size, 30.)
            self.posterior_noise, self.posterior_outputscale, self.posterior_lengthscale = self.draw_prior_hyperparams()

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

        def hyperparam_log_prior(self, prior_mode: bool = False):
            self.select_hyperparameters(prior_mode)
            return self.noise_var_prior.log_prob(self.likelihood.noise) + \
                   self.outscale_prior.log_prob(self.covar_module.outputscale) + \
                   self.lscale_prior.log_prob(self.covar_module.base_kernel.lengthscale)

        def draw_prior_hyperparams(self):
            return self.noise_var_prior.sample(), self.outscale_prior.sample(), self.lscale_prior.sample()

        def select_hyperparameters(self, prior_mode: bool = False):
            if prior_mode:
                # self.likelihood.noise, self.covar_module.outputscale, self.covar_module.base_kernel.lengthscale = \
                #     self.draw_prior_hyperparams()
                self.likelihood.noise = self.noise_var_prior.mean
                self.covar_module.outputscale = self.outscale_prior.mean
                self.covar_module.base_kernel.lengthscale = self.lscale_prior.mean
            else:
                self.likelihood.noise = self.posterior_noise
                self.covar_module.outputscale = self.posterior_outputscale
                self.covar_module.base_kernel.lengthscale = self.posterior_lengthscale

        def param_dict(self):
            params = {'posterior_noise': self.posterior_noise,
                      'posterior_outputscale': self.posterior_outputscale,
                      'posterior_lengthscale': self.posterior_lengthscale}
            return params

        def load_param_dict(self, param_dict):
            self.posterior_noise = param_dict['posterior_noise']
            self.posterior_outputscale = param_dict['posterior_outputscale']
            self.posterior_lengthscale = param_dict['posterior_lengthscale']

    class ExactGPModelLinearKernel(gpytorch.models.ExactGP):
        # ATTENTION: do not name the HP priors "noise_prior", "outputscale_prior" or "lengthscale_prior"
        noise_var_prior = dist.Gamma(50., 500.)
        outscale_prior = dist.Gamma(100., 10.)

        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.LinearKernel()

            # draw kernel parameters
            self.posterior_noise, self.posterior_outputscale = self.draw_prior_hyperparams()

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

        def hyperparam_log_prior(self, prior_mode: bool = False):
            self.select_hyperparameters(prior_mode)
            return self.noise_var_prior.log_prob(self.likelihood.noise) + \
                   self.outscale_prior.log_prob(self.covar_module.variance)

        def draw_prior_hyperparams(self):
            return self.noise_var_prior.sample(), self.outscale_prior.sample()

        def select_hyperparameters(self, prior_mode: bool = False):
            if prior_mode:
                # self.likelihood.noise, self.gp.covar_module.outputscale = \
                #     self.draw_prior_hyperparams()
                self.likelihood.noise = self.noise_var_prior.mean
                self.covar_module.outputscale = self.outscale_prior.mean
            else:
                self.likelihood.noise = self.posterior_noise
                self.covar_module.outputscale = self.posterior_outputscale

        def param_dict(self):
            params = {'posterior_noise': self.posterior_noise,
                      'posterior_outputscale': self.posterior_outputscale}
            return params

        def load_param_dict(self, param_dict):
            self.posterior_noise = param_dict['posterior_noise']
            self.posterior_outputscale = param_dict['posterior_outputscale']

    def __init__(self, in_size: int, static=False, linear=False):
        super().__init__(in_size)

        # initialize likelihood and gp model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if linear:
            self.gp = GaussianProcess.ExactGPModelLinearKernel(None, None, likelihood)
        else:
            self.gp = GaussianProcess.ExactGPModelRQKernel(None, None, likelihood, in_size)
        self.static = static
        self.linear = linear

        # can set true kernel parameters for testing
        # self.gp.likelihood.noise = 0.1
        # self.gp.covar_module.outputscale = 10.
        # self.gp.covar_module.base_kernel.lengthscale = self.in_size

        if static:
            self.init_as_static()

    def init_as_static(self):
        # generate support points and sample training targets from the GP prior
        num_train = 50 * self.in_size
        train_x = 20. * (torch.rand((num_train, self.in_size)) - 0.5)
        self.eval()
        with gpytorch.settings.prior_mode(True):
            self.gp.select_hyperparameters(prior_mode=False)
            f_dist = self.gp(train_x)
            y_dist = self.gp.likelihood(f_dist)
            train_y = y_dist.sample().detach()

        # update GP data
        self.set_data(train_x, train_y)

    def set_data(self, inputs: torch.Tensor, targets: torch.Tensor):
        self._check_args(inputs, targets)
        self.gp.set_train_data(inputs, targets, strict=False)

    def forward(self, inputs: torch.Tensor, prior_mode=False):
        self._check_args(inputs)
        output_shape = (*inputs.shape[:-1], 1)

        self.eval()
        with gpytorch.settings.prior_mode(prior_mode):
            self.gp.select_hyperparameters(prior_mode)
            f_dist = self.gp(inputs)
        return f_dist.mean.view(output_shape)

    def sample(self, inputs: torch.Tensor, prior_mode=False):
        self._check_args(inputs)
        output_shape = (*inputs.shape[:-1], 1)

        self.eval()
        with gpytorch.settings.prior_mode(prior_mode):
            self.gp.select_hyperparameters(prior_mode)
            f_dist = self.gp(inputs)
        y_dist = self.gp.likelihood(f_dist.mean) if self.static else self.gp.likelihood(f_dist)
        return y_dist.sample().view(output_shape)

    def mll(self, inputs: torch.Tensor, targets: torch.Tensor, prior_mode=False, reduce=True):
        self._check_args(inputs, targets)
        output_shape = targets.shape[:-1]
        with gpytorch.settings.prior_mode(prior_mode):
            self.gp.select_hyperparameters(prior_mode)
            f_dist = self.gp(inputs)

        if self.static:
            y_dist = self.gp.likelihood(f_dist.mean)
            mlls = y_dist.log_prob(targets).squeeze(-1)
        else:
            y_dist = self.gp.likelihood(f_dist)
            mlls = y_dist.log_prob(targets)
        assert mlls.shape == output_shape, print(f'Invalid shape {mlls.shape}!')

        if reduce:
            return mlls.sum()
        return mlls

    def expected_noise_entropy(self, prior_mode: bool = False) -> torch.Tensor:
        # use point estimate with the MAP variance
        self.gp.select_hyperparameters(prior_mode)
        return 0.5 * (2. * math.pi * self.gp.likelihood.noise * math.e).log().squeeze()

    def param_dict(self):
        gp_param_dict = self.gp.param_dict()
        params = {'in_size': self.in_size,
                  'static': self.static,
                  'linear': self.linear,
                  'gp_param_dict': gp_param_dict}

        if self.static:
            params['train_inputs'] = self.gp.train_inputs
            params['train_targets'] = self.gp.train_targets
        return params

    def load_param_dict(self, param_dict):
        self.in_size = param_dict['in_size']
        self.static = param_dict['static']
        self.linear = param_dict['linear']
        self.gp.load_param_dict(param_dict['gp_param_dict'])

        if self.static:
            self.gp.train_inputs = param_dict['train_inputs']
            self.gp.train_targets = param_dict['train_targets']
