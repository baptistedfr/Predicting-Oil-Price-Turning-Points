
import plotly.graph_objects as go
from typing import Tuple
from GQLib.Models import LPPL
import scipy.stats
import numpy as np

from .abstract_optimizer import Optimizer

class MCMC(Optimizer):
    """
    MCMC optimizer class based on Metropolis Hastings algorithm, i.e. bayesian statistics.
    The algorithm is based on two principle : 
        - Monte Carlo simulations : simulating a large number of random variables and compute the expectation as a result of the estimation
        - Markov Chains : each iteration of the parameters sequence is linked to the previous estimation

    As it is a bayesian algorithm, the model computes the posterior distribution as the product of a prior distribution and a likelihood function.
    Thus, we need to define prior distributions for the parameters (here considered as gaussian distributions) and likelihood (here RSS).
    Every iteration is based on the previous as the proposal value for the next iteration is sampled from a gaussian distribution centered on the previous value.
    Then, the proposal value is accepted or not as the new iteration by calculating an acceptance ratio and comparing it to a uniform distribution.
    """

    def __init__(self, initial_individual: np.ndarray) -> None:
        """
        Initialize the optimizer with a specific frequency.

        Parameters
        ----------
        initial_individual : np.ndarray
            Parameters initial values
        """
        self.NB_ITERATION = None
        self.PROPOSAL_STD = None
        self.BURNIN_PERIOD = None
        self.initial_individual = initial_individual

    def log_likelihood(self, individual: np.ndarray , data: np.ndarray) -> float:
        """
        Compute the likelihood = the fitness of a individual.

        Parameters
        ----------
        individual : np.ndarray
            Array representing the LPPL parameters [t_c, alpha, omega, phi].
        data : np.ndarray
            A 2D array of shape (J, 2), where:
                - Column 0 is time.
                - Column 1 is the observed price.
        """
        return -LPPL.numba_RSS(individual, data)/2

    def propose_individual(self, current_individual: np.ndarray) -> np.ndarray:
        """
        Sample a new individual based on a gaussian distribution centered around the current individual values.

        Parameters
        ----------
        current_individual : np.ndarray
            Array representing the LPPL parameters [t_c, alpha, omega, phi] of the previous iteration.
        """
        return np.random.normal(current_individual, self.PROPOSAL_STD)
    
    def proposal_density(self, quantile_individual: np.ndarray, centered_individual: np.ndarray) -> float:
        """
        Compute the proposal density for a given quantile level centered on a given individual.

        Parameters
        ----------
        centered_individual : np.ndarray
            Compute the distribution centered on this point;
        quantile_individual : np.ndarray
            Compute the distribution at the point.
        """
        return np.prod(scipy.stats.norm.pdf(quantile_individual, loc=centered_individual, scale=self.PROPOSAL_STD))

    def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Fit the model parameters to a given subinterval of the data with Metropolis-Hastings algorithm0

        Parameters
        ----------
        start : int
            The start index of the subinterval.
        end : int
            The end index of the subinterval.
        data : np.ndarray
            A 2D array with time in the first column and observed values in the second.

        Returns
        -------
        Tuple[float, np.ndarray]
            The best fitness value (RSS) and the corresponding parameter set.
        """
        self.fitness_history = []
        self.individual_history = []

        # Initialize the first individual (first set of parameters) and it's likelihood
        current_individual = self.initial_individual
        current_likelihood = self.log_likelihood(current_individual, data)
        self.individual_history.append(current_individual)
        self.fitness_history.append(abs(current_likelihood))

        for _ in range(self.NB_ITERATION):
            
            # Draw of a new individual from the proposal distribution
            proposed_individual = self.propose_individual(current_individual)

            # Likelihood calculation
            proposed_likelihood = self.log_likelihood(proposed_individual, data)

            # Prior distributions
            current_prior = self.proposal_density(current_individual, proposed_individual)
            proposal_prior = self.proposal_density(proposed_individual, current_individual)

            # Acceptance ratio calculation based on likelihoods
            acceptance_ratio = min(np.exp(proposed_likelihood - current_likelihood) * proposal_prior / current_prior, 1)
            
            # Acceptance or rejection based on a uniform random variable
            if np.random.rand() < acceptance_ratio:
                # Acceptance
                current_individual = proposed_individual
                current_likelihood = proposed_likelihood
            else:
                # Rejection
                current_individual = current_individual

            # Fitness and parameters save in memory
            self.fitness_history.append(abs(current_likelihood))
            self.individual_history.append(current_individual)

        # The MCMC optimized parameters are the mean (expectation of Monte Carlo) of each parameter after the burn-in period
        after_burnin_period = self.individual_history[self.BURNIN_PERIOD:]
        expectation = np.mean(after_burnin_period, axis=0)

        best_fitness = self.fitness_history[-1]

        return (best_fitness, expectation)
    
    def plot_parameter_evolution(self) -> None:
        """
        Plot the evolution of each parameter over iterations using subplots.

        Parameters
        ----------
        parameter_history : List[np.ndarray]
            List of parameter sets over iterations (one array per iteration).
        """
        # Convert the list of parameter arrays into a 2D array (iterations x parameters)
        parameter_array = np.array(self.individual_history)
        num_parameters = parameter_array.shape[1]
        param_names = ["tc","alpha", "omega", "phi"]

        # Create subplots, one for each parameter
        for i in range(num_parameters):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.individual_history))),
                    y=parameter_array[:, i],
                    mode='lines',
                    name=f'Parameter {param_names[i]}'
                )
            )
            fig.update_layout(
                title="Evolution of Parameters Over Iterations",
                xaxis_title="Iterations",
                yaxis_title="Parameter Values",
                showlegend=True,
            )
            fig.show()
