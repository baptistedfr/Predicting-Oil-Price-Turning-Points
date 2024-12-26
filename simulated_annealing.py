import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class LPPLSimulatedAnnealing:
    def __init__(self, max_iterations=1000, initial_temp=1.0, cooling_rate=0.999, bounds=None):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.bounds = bounds or [
            (1, 2520),    # tc
            (0.1, 0.9),   # alpha
            (0, 40),      # omega
            (0, 2 * np.pi) # phi
        ]

    def lppl(self, t, A, B, tc, alpha, C, omega, phi):
        return A + B * (tc - t) ** alpha + C * (tc - t) ** alpha * np.cos(omega * np.log(tc - t) + phi)

    def fitness_function(self, params, t, P_actual):
        A, B, C, tc, alpha, omega, phi = params
        try:
            P_predicted = self.lppl(t, A, B, tc, alpha, C, omega, phi)
            return float(np.sum((P_actual - P_predicted) ** 2))
        except Exception:
            return 1e10  # High error for invalid parameters

    def fit_lppl(self, data, params, P_actual):
        y = data.iloc[:, 1].values
        t = data.iloc[:, 0].values

        tc, alpha, omega, phi = params

        dt = np.abs(tc - t)
        if np.any(dt <= 0):
            return np.inf

        f = dt ** alpha
        g = f * np.cos(omega * np.log(dt) + phi)

        # Construction de la matrice de régression linéaire
        V = np.column_stack((np.ones_like(f), f, g))

        try:
            # Résolution du système linéaire pour trouver A, B, C
            params_lin = np.linalg.inv(V.T @ V) @ (V.T @ y)
            A, B, C = params_lin[0], params_lin[1], params_lin[2]
        except np.linalg.LinAlgError:  # Si la matrice est singulière
            return np.inf

        # Reconstruction des paramètres complets
        all_params = [float(A), float(B), float(C)] + [float(p) for p in params]

        # Évaluation de la fonction d'erreur (RSS)
        try:
            rss = self.fitness_function(all_params, t, P_actual)
            return rss
        except Exception as e:
            return np.inf

    def generate_initial_solution(self):
        return [np.random.uniform(low, high) for (low, high) in self.bounds]

    def perturb_solution(self, solution):
        """
        Perturb all parameters in the solution by sampling them from their respective bounds.

        Args:
        - solution (list): the current solution with parameter values

        Returns:
        - perturbed (list): the new solution with perturbed parameters
        """
        perturbed = solution[:]

        # Perturb each parameter independently by sampling from the bounds
        for i in range(len(solution)):
            low, high = self.bounds[i]
            perturbed[i] = np.random.uniform(low, high)  # Uniform sampling within bounds

        return perturbed

    def run(self, data):
        t = np.arange(len(data))
        P_actual = data["Price"].values
        data = pd.concat([pd.Series(t), pd.Series(P_actual)], axis=1)

        # Initial solution
        current_solution = self.generate_initial_solution()
        best_solution = current_solution[:]
        current_fitness = self.fit_lppl(data, current_solution, P_actual)
        best_fitness = current_fitness

        # Temperature initialization
        temperature = self.initial_temp

        for iteration in range(self.max_iterations):
            # Generate a new candidate solution
            candidate_solution = self.perturb_solution(current_solution)
            candidate_fitness = self.fit_lppl(data, candidate_solution, P_actual)

            # Acceptance probability
            if candidate_fitness < current_fitness:
                accept = True
            else:
                delta = candidate_fitness - current_fitness
                accept = random.random() < np.exp(-delta / temperature)

            # Accept or reject the candidate solution
            if accept:
                current_solution = candidate_solution
                current_fitness = candidate_fitness

            # Update the best solution found so far
            if current_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness

            # Cooling schedule
            temperature *= self.cooling_rate
            if temperature < 1e-3:
                break

            # Logging progress
            if iteration % 100 == 0 or iteration == self.max_iterations - 1:
                print(f"Iteration {iteration}: Best Fitness = {best_fitness}")

        return best_solution, best_fitness

if __name__ == "__main__":
    # Fixer les graines pour numpy et random
    np.random.seed(123)
    random.seed(123)

    # Chargement des données
    data = pd.read_csv("WTI_Spot_Price_daily.csv", sep=";")
    data = data.iloc[4:, :]
    data[['Date', 'Price']] = data['Cushing OK WTI Spot Price FOB'].str.split(',', expand=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Price'] = pd.to_numeric(data['Price'])
    data.set_index('Date', inplace=True)
    data.drop(columns=['Cushing OK WTI Spot Price FOB'], inplace=True)
    data.sort_index(inplace=True)

    # Test avec une portion des données
    test_data = data.loc["2004-01-01":"2008-01-01"]

    sa = LPPLSimulatedAnnealing()
    best_solution, best_fitness = sa.run(test_data)

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

    turning_point = int(best_solution[0])  # tc (turning point)

    # Localisation du turning point par rapport aux indices globaux
    test_start_index = data.index.get_loc(test_data.index[-1])

    # Calculez l'indice global du turning point
    global_turning_point_index = test_start_index + turning_point

    # Tracez le graphique
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Price"], label="Prix réel", color="blue")
    plt.axvline(x=data.index[global_turning_point_index], color="red", linestyle="--", label="Turning Point Prédit (tc)")
    plt.title("Prix réel et Turning Point Prédit")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.show()
