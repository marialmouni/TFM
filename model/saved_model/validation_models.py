import numpy as np
import random
import pandas as pd
import os
from model.valid_model.validate_dqn import validate_dqn

def random_policy(env, num_assets):
    """
    Política aleatoria para una cartera de activos: toma decisiones aleatorias para cada activo.
    """
    total_reward = []
    done = False
    obs = env.reset()  # Restablecer el entorno y obtener el primer estado

    while not done:
        # Genera una acción aleatoria para cada activo
        action = np.array([random.choice(range(3)) for _ in range(num_assets)])  # Acciones para cada activo
        next_obs, reward, done = env.step(action)  # Pasa las acciones como un vector al entorno
        total_reward += reward
    
    return total_reward


def always_buy_policy(env):
    """
    Estrategia basada en reglas: siempre comprar para cada activo.
    """
    total_reward = []
    done = False
    obs = env.reset()
    
    while not done:
        # Crear una lista de acciones de compra (1) para todos los activos
        actions = [1] * env.num_assets  # Acción de compra para cada activo
        next_obs, reward, done = env.step(actions)
        total_reward += reward
    
    return total_reward


def always_sell_policy(env):
    """
    Estrategia basada en reglas: siempre vender.
    """
    total_reward = []
    done = False
    obs = env.reset()
    
    while not done:
        action = [2] * env.num_assets  # Acción de venta
        next_obs, reward, done = env.step(action)
        total_reward += reward
    
    return total_reward

def always_hold_policy(env):
    """
    Estrategia basada en reglas: siempre mantener.
    """
    total_reward = []
    done = False
    obs = env.reset()
    
    while not done:
        action = [0] * env.num_assets # Acción de mantener
        next_obs, reward, done = env.step(action)
        total_reward += reward
    
    return total_reward

def compare_to_baseline(env, trained_model, save_path=None):
    test_rewards = validate_dqn(env=env, Q=trained_model)
    avg_test_reward = np.mean(test_rewards)
    # Baseline: Aleatorio
    random_rewards = [random_policy(env, env.num_assets) for _ in range(100)]
    avg_random_reward = np.mean(random_rewards)
    # Baseline: Siempre compra
    always_buy_rewards = [always_buy_policy(env) for _ in range(100)]
    avg_always_buy_reward = np.mean(always_buy_rewards)
    # Baseline: Siempre vende
    always_sell_rewards = [always_sell_policy(env) for _ in range(100)]
    avg_always_sell_reward = np.mean(always_sell_rewards)
    # Baseline: Siempre mantiene
    always_hold_rewards = [always_hold_policy(env) for _ in range(100)]
    avg_always_hold_reward = np.mean(always_hold_rewards)

    # Crear un DataFrame con los resultados
    validation_data = pd.DataFrame({
        'Model': ['Trained', 'Random', 'Always Buy', 'Always Sell', 'Always Hold'],
        'Average Reward': [
            avg_test_reward, avg_random_reward, 
            avg_always_buy_reward, avg_always_sell_reward, avg_always_hold_reward
        ]
    })

    # Guardar en CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    validation_data.to_csv(save_path, index=False)
    print(f"Resultados guardados en {save_path}.")

def print_validation_data(data):
    """
    Imprime los resultados de validación desde un DataFrame o CSV.
    """
    if isinstance(data, str):  # Si el argumento es un path, cargar el CSV
        data = pd.read_csv(data)
    print("\nResultados de validación:")
    print(data)

