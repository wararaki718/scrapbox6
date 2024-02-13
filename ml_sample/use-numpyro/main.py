import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS


def eight_schools(J, sigma, y=None) -> None:
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)


def main() -> None:
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    kernel = NUTS(eight_schools)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, J, sigma, y=y, extra_fields=('potential_energy',))
    mcmc.print_summary()

    pe = mcmc.get_extra_fields()['potential_energy']
    print("Expected log joint density: {:.2f}".format(-pe.mean()))
    print("DONE")


if __name__ == "__main__":
    main()
