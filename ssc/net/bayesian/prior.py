import math
import torch
import numpy as np
import torch.nn as nn

"""
All prior distributions center around 0.
"""

eps = float(np.finfo(float).eps)

class Prior(nn.Module):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__.lower()] = cls

    @classmethod
    def create(cls, prior_type, params):
        if prior_type not in cls.subclasses:
            raise ValueError('Bad prior type {}'.format(prior_type))

        return cls.subclasses[prior_type](**params)

    @classmethod
    def create_from_cfg(cls, cfg):
        if 'prior' not in cfg:
            return Dummy()
        class_name = cfg['prior']['name'].lower()
        params = {p:cfg['prior'][p] for p in cfg['prior'].keys() - set(['name'])}
        return cls.create(class_name, params)

class Gaussian(Prior):
    def __init__(self, sigma = 1):
        super().__init__()
        self.sigma = sigma

    # Assume KL divergence with gaussian distributions
    def kl_div(self, w, g_mu, g_sigma):
        kl = math.log(self.sigma) - torch.log(g_sigma) + (g_sigma**2 + g_mu**2) / (eps + 2 * self.sigma ** 2) - 0.5
        return kl.sum()

def gauss_single_logp(x, sigma):
    return -0.5*(x.numel()*math.log(2*math.pi*sigma**2) + torch.sum((x/(eps + sigma))**2))

def gauss_single_p(x, sigma):
    return torch.exp(-.5*(x/(eps + sigma))**2)/math.sqrt(2*math.pi*sigma**2)

def gauss_multi_logp(x, mu, sigma):
    return -0.5*torch.sum(torch.log(2*math.pi*sigma**2) + ((x-mu)/(eps + sigma))**2)

class GaussianMixture(Prior):
    def __init__(self, sigma0 = 1, sigma1 = 10, alpha = 0.5):
        super().__init__()
        self.log_prob = lambda x: torch.log(alpha*gauss_single_p(x,sigma0) + (1.-alpha)*gauss_single_p(x,sigma1)).sum()

    # Assume KL divergence with gaussian distributions
    def kl_div(self, w, g_mu, g_sigma):
        posterior = gauss_multi_logp(w, g_mu, g_sigma)
        prior = self.log_prob(w)
        return posterior - prior

class Cauchy(Prior):
    def __init__(self, gamma = 1):
        super().__init__()
        self.log_prob = lambda x: - x.numel()*math.log(math.pi*gamma) - torch.log(1 + (x/gamma)**2).sum()

    # Assume KL divergence with gaussian distributions
    def kl_div(self, w, g_mu, g_sigma):
        posterior = gauss_multi_logp(w, g_mu, g_sigma)
        prior = self.log_prob(w)
        return posterior - prior

class Dummy(Prior):
    # Assume KL divergence with gaussian distributions
    def kl_div(self, w, g_mu, g_sigma):
        if self.train:
            raise NotImplementedError('Prior must be set for training')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.distributions.cauchy import Cauchy as tCauchy
    from torch.distributions.normal import Normal as tNormal
    import time

    x = torch.linspace(-0.1, 0.1, steps=100)


    c1 = Cauchy(1).log_prob(x)
    c2 = tCauchy(0,1).log_prob(x).sum()
    n1 = gauss_single_logp(x, 1)
    n2 = tNormal(0,1).log_prob(x).sum()

    print(c1, c2, n1, n2)

    sigma = torch.linspace(0.9, 1.1, steps=100)
    w = tNormal(x, sigma).sample()

    kl1 = Gaussian().kl_div(w, x, sigma)
    alpha = 0.8
    t = time.time()
    kl2 = GaussianMixture(alpha=alpha).kl_div(w, x, sigma)
    print('Mixture took', time.time() - t)
    t = time.time()
    kl3 = Cauchy(1).kl_div(w, x, sigma)
    t = time.time()
    print('cauchy took', time.time() - t)
    print('Gaussian KL', kl1)
    print('Gaussian Mixture KL, (alpha: {})'.format(alpha), kl2)
    print('Cauchy KL', kl3)

    x_plot = torch.linspace(-10, 10, steps=100)
    x_npy = x_plot.numpy()
    plt.figure()
    plt.plot(x_npy, tNormal(0,1).log_prob(x_plot).exp().numpy(), label = 'Normal')

    plt.plot(x_npy, alpha*tNormal(0,1).log_prob(x_plot).exp().numpy()+(1-alpha)*tNormal(0,10).log_prob(x_plot).exp().numpy() , label = 'Gaussmixture: {}'.format(kl2))
    plt.plot(x_npy, tCauchy(0,1).log_prob(x_plot).exp().numpy(), label = 'Cauchy: {}'.format(kl3))
    plt.legend()
    plt.savefig('Dist.png')
    plt.close()
