import numpy as np
from scipy.optimize import minimize

def markowitz_parameters(expected_returns, cov_matrix, alpha_1, alpha_2, beta_1, beta_2):
    """
    Calcula parámetros para el modelo de Markowitz.

    Args:
        expected_returns (np.ndarray): Vector 1D con las rentabilidades esperadas de los activos.
        cov_matrix (np.ndarray): Matriz 2D de covarianza de los activos.
        alpha_1 (float): Factor de ajuste para el límite inferior de r_min.
        alpha_2 (float): Factor de ajuste para el límite superior de r_min.
        beta_1 (float): Factor de ajuste para el límite inferior de sigma_max.
        beta_2 (float): Factor de ajuste para el límite superior de sigma_max.

    Returns:
        tuple: Valores aleatorios para r_min y sigma_max seleccionados dentro de sus respectivos intervalos.
    """
    # Calcula los valores mínimos y máximos de las rentabilidades esperadas
    mu_min = np.min(expected_returns)
    mu_max = np.max(expected_returns)

    # Define el intervalo para r_min
    r_min_lower = mu_min + alpha_1 * (mu_max - mu_min)
    r_min_upper = mu_max - alpha_2 * (mu_max - mu_min)

    # Selecciona un valor aleatorio para r_min dentro de su intervalo
    r_min = np.random.uniform(r_min_lower, r_min_upper)

    # Calcula las volatilidades mínima y máxima
    volatilities = np.sqrt(np.diag(cov_matrix))
    sigma_min = np.min(volatilities)
    sigma_max = np.max(volatilities)

    # Define el intervalo para sigma_max
    sigma_max_lower = sigma_min + beta_1 * (sigma_max - sigma_min)
    sigma_max_upper = sigma_max - beta_2 * (sigma_max - sigma_min)

    # Selecciona un valor aleatorio para sigma_max dentro de su intervalo
    sigma_max = np.random.uniform(sigma_max_lower, sigma_max_upper)

    return r_min, sigma_max

def markowitz_weights(returns, cov_matrix, r_min, sigma_max):
    """
    Calcula los pesos óptimos de una cartera siguiendo el modelo de Markowitz.

    Args:
        returns (numpy.ndarray): Rentabilidades esperadas de los activos (vector 1D de tamaño N).
        cov_matrix (numpy.ndarray): Matriz de covarianza de los activos (matriz 2D de NxN).
        r_min (float): Rentabilidad mínima requerida de la cartera.
        sigma_max (float): Volatilidad máxima permitida de la cartera.

    Returns:
        numpy.ndarray: Pesos óptimos para los activos.
    """
    num_assets = len(returns)

    # Restricción: La suma de los pesos debe ser 1
    def weight_sum_constraint(weights):
        return np.sum(weights) - 1

    # Restricción: Rentabilidad mínima
    def min_return_constraint(weights):
        return np.dot(weights, returns) - r_min

    # Restricción: Volatilidad máxima
    def max_volatility_constraint(weights):
        return sigma_max - np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Función objetivo: penalizar el incumplimiento de restricciones
    def objective_function(weights):
        expected_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        penalty_return = max(0, r_min - expected_return)  # Penalizar si no se cumple el retorno mínimo
        penalty_volatility = max(0, portfolio_volatility - sigma_max)  # Penalizar si supera la volatilidad máxima
        return penalty_return + penalty_volatility

    # Inicialización de pesos iguales
    initial_weights = np.ones(num_assets) / num_assets

    # Fronteras de los pesos: deben estar entre 0 y 1
    bounds = [(0, 1) for _ in range(num_assets)]

    # Restricciones
    constraints = [
        {"type": "eq", "fun": weight_sum_constraint},
        {"type": "ineq", "fun": min_return_constraint},
        {"type": "ineq", "fun": max_volatility_constraint},
    ]

    # Resolver el problema de optimización
    result = minimize(
        objective_function,
        initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    if result.success:
        return result.x  # Pesos óptimos
    else:
        raise ValueError("La optimización no convergió: " + result.message)
