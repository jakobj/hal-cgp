import functools
import math
import multiprocessing as mp
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..individual import IndividualBase


class EvolutionStrategies:
    def __init__(
        self,
        objective: Callable[[IndividualBase], float],
        seed: int,
        *,
        learning_rate_mu: float = 1.0,
        learning_rate_sigma: Union[None, float] = None,
        population_size: Union[None, int] = None,
        max_generations: int = 10,
        min_sigma: float = 1e-6,
        fitness_shaping: bool = True,
        mirrored_sampling: bool = True,
        n_processes: int = 1
    ) -> None:
        """Evolution strategies using the natural gradient of multinormal
        search distributions in natural coordinates.

        Standard deviation of the search distribution for each
        parameter is stored per individual. Offspring may use the
        values of their parents if the number of used parameters has
        not changed. Does not consider covariances between parameters.

        Implementation and default values following Wierstra et
        al. (2014). Natural evolution strategies. Journal of Machine
        Learning Research, 15(1), 949-980.

        Parameters
        ----------
        individual : Individual
            Individual for which to perform local search.
        objective : Callable[[Callable[[np.ndarray[float]], np.ndarray[float]]], float]
            Objective function used for the local search. Needs to accept
            an numpy-compatible function as first argument (e.g., from
            IndividualBase.to_numpy()) and return a float representing the
            fitness of the individual.
        seed : int
            Seed for internal random number generator.
        learning_rate_mu : float, optional
            Learning rate for mean of search distribution.
            Defaults to 1.0.
        learning_rate_sigma : float, optional
            Learning rate for standard deviation of search
            distribution. Defaults to (3 + log(dim)) / (5.0 *
            sqrt(dim)), where dim is the dimension of the search space.
        population_size : int, optional
            Number of samples evaluated per generation. Defaults to 4
            + floor(3 * log(dim))), where dim is the dimension of the
            search space..
        max_generations : int, optional
            Maximal number of generations. Defaults to 10.
        min_sigma : float, optional
            Minimal value for standard deviation of search
            distribution. Defaults to 1e-6.
        fitness_shaping : bool, optional
            Whether to use fitness shaping. Defaults to True.
        mirrored_sampling : bool, optional
            Whether to use mirrored sampling. WARNING: Doubles the number
            of samples per generation. Defaults to True.
        n_processes : int, optional
            Number of parallel processes. Defaults to 1.

        """
        self.objective = objective
        self.seed = seed
        self.learning_rate_mu = learning_rate_mu
        self.learning_rate_sigma = learning_rate_sigma
        self.population_size = population_size
        self.max_generations = max_generations
        self.min_sigma = min_sigma
        self.fitness_shaping = fitness_shaping
        self.mirrored_sampling = mirrored_sampling
        self.n_processes = n_processes

        self.sigma: Dict[int, np.ndarray[float]] = {}

    def __call__(self, ind: "IndividualBase") -> None:

        process_pool: Union[None, mp.pool.Pool]
        if self.n_processes > 1:
            process_pool = mp.Pool(processes=self.n_processes)
        else:
            process_pool = None

        rng = np.random.RandomState(self.seed)

        mu: np.ndarray[float]
        params_names: List[str]
        mu, params_names = ind.parameters_to_numpy_array(only_active_nodes=True)

        sigma: np.ndarray[float]
        if ind.idx in self.sigma:  # try loading sigma
            sigma = self.sigma[ind.idx]
            assert len(sigma) == len(mu)
        elif ind.parent_idx in self.sigma and len(mu) == len(
            self.sigma[ind.parent_idx]
        ):  # try loading sigma from parent
            sigma = self.sigma[ind.parent_idx]
        else:
            # as a simple heuristic initialize sigma to 10% of the
            # mean value
            sigma = 0.1 * np.abs(mu.copy())

        if len(mu) > 0:
            if self.learning_rate_sigma is None:
                learning_rate_sigma = (3 + math.log(len(mu))) / (5.0 * math.sqrt(len(mu)))
            else:
                learning_rate_sigma = self.learning_rate_sigma

            if self.population_size is None:
                population_size = 4 + int(math.floor(3 * math.log(len(mu))))
            else:
                population_size = self.population_size

            obj: Callable[[np.ndarray[float]], float] = functools.partial(
                self._objective_wrapper, ind=ind, params_names=params_names
            )

            for generation in range(self.max_generations):

                s: np.ndarray[float]
                z: np.ndarray[float]
                s, z = self._sample_s_and_z(mu, sigma, population_size, rng)

                fitness: np.ndarray[float]
                if self.n_processes == 1:
                    fitness = np.fromiter(map(obj, z), np.float)
                else:
                    assert isinstance(process_pool, mp.pool.Pool)
                    fitness = np.fromiter(process_pool.map(obj, z), np.float)

                s, z, utility = self._determine_utility(s, z, fitness)

                mu, sigma = self._update_parameters_of_search_distribution(
                    s, utility, learning_rate_sigma, mu, sigma
                )

                if np.all(sigma < self.min_sigma):
                    break
                sigma[sigma < self.min_sigma] = self.min_sigma

            ind.update_parameters_from_numpy_array(mu, params_names)

        # WARNING: should not be indented as all individuals should
        # store their sigma whether they have changed or not to allow
        # propagation over many generations
        assert isinstance(ind.idx, int)
        self.sigma[ind.idx] = sigma.copy()

        if self.n_processes > 1:
            assert process_pool is not None
            process_pool.close()

    def _sample_s_and_z(
        self, mu: np.ndarray, sigma: np.ndarray, population_size: int, rng: "np.random.RandomState"
    ) -> Tuple["np.ndarray[float]", "np.ndarray[float]"]:
        s: np.ndarray[float] = rng.normal(0, 1, size=(population_size, *mu.shape))
        z: np.ndarray[float] = mu + sigma * s

        if self.mirrored_sampling:
            z = np.vstack([z, mu - sigma * s])
            s = np.vstack([s, -s])

        return s, z

    def _objective_wrapper(
        self, z: "np.ndarray[float]", *, ind: "IndividualBase", params_names: List[str]
    ) -> float:
        new_ind: IndividualBase = ind.clone()
        new_ind.update_parameters_from_numpy_array(z, params_names)
        return self.objective(new_ind)

    def _determine_utility(
        self, s: "np.ndarray[float]", z: "np.ndarray[float]", fitness: "np.ndarray[float]",
    ) -> Tuple["np.ndarray[float]", "np.ndarray[float]", "np.ndarray[float]"]:
        if self.fitness_shaping:
            order, utility = self._utility_function(fitness)
            s = s[order]
            z = z[order]
        else:
            utility = fitness

        return s, z, utility

    def _utility_function(
        self, fitness: np.ndarray
    ) -> Tuple["np.ndarray[int]", "np.ndarray[float]"]:
        n: int = len(fitness)
        order: np.ndarray[float] = np.argsort(fitness)[::-1]

        fitness = fitness[order]

        utility: np.ndarray[float] = [
            np.max([0, math.log((n / 2) + 1)]) - math.log(k + 1) for k in range(n)
        ]
        utility = utility / np.sum(utility) - 1.0 / n

        return order, utility

    def _update_parameters_of_search_distribution(
        self,
        s: np.ndarray,
        utility: "np.ndarray[float]",
        learning_rate_sigma,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> Tuple["np.ndarray[float]", "np.ndarray[float]"]:
        mu += self.learning_rate_mu * sigma * np.dot(utility, s)
        sigma *= np.exp(learning_rate_sigma / 2.0 * np.dot(utility, s ** 2 - 1))
        return mu, sigma
