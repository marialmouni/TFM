import tensorflow as tf
import os
import pandas as pd
import numpy as np
import sys
from utils.data_loader import load_data
from utils.preprocess import preprocess_data
from utils.markowitz_model import markowitz_parameters, markowitz_weights
from model.environment import Environment
from model.Q_network import Q_Network
from model.train_dqn import train_dqn
from model.valid_model.validation_models import compare_to_baseline, print_validation_data
from utils.recommend import daily_recommendation

def main():
    # Leer el argumento para seleccionar la simulación
    if len(sys.argv) < 2:
        print("Por favor, proporciona el número de la simulación como argumento.")
        sys.exit(1)

    simulation_number = sys.argv[1]
    simulation_file = f"simulations/simulation{simulation_number}_data.csv"

    if not os.path.exists(simulation_file):
        print(f"Archivo de simulación no encontrado: {simulation_file}")
        sys.exit(1)

    # Cargar configuración de la simulación
    simulation_data = pd.read_csv(simulation_file)
    SYMBOLS = eval(simulation_data['symbols'].iloc[0])  # Convertir la cadena a lista
    START_DATE = simulation_data['start_date'].iloc[0]
    INITIAL_CAPITAL = simulation_data['initial_capital'].iloc[0]
    N_EPISODES = simulation_data['N_episodes'].iloc[0]
    BATCH_SIZE = simulation_data['batch_size'].iloc[0]
    ALPHA_1 = simulation_data['alpha_1'].iloc[0]
    ALPHA_2 = simulation_data['alpha_2'].iloc[0]
    BETA_1 = simulation_data['beta_1'].iloc[0]
    BETA_2 = simulation_data['beta_2'].iloc[0]

    DATA_PATH = f"data/raw/historical_data_{simulation_number}.csv"
    PROCESSED_DATA_PATH = f"data/processed/synchronized_data_{simulation_number}.csv"
    MODEL_PATH = f"model/saved_model/trained_model_{simulation_number}.keras"
    REWARDS_PATH = f"model/saved_model/trained_model_rewards{simulation_number}.txt"
    VALIDATION_PATH = f"model/valid_model/validation_data_{simulation_number}.csv"

    # 1. Cargar datos
    if not os.path.exists(DATA_PATH):
        print("Descargando datos...")
        data = load_data(SYMBOLS, START_DATE, save_path=DATA_PATH)
    else:
        print("Cargando datos desde archivo...")
        data = pd.read_csv(DATA_PATH)

    # 2. Procesar datos
    print("Procesando datos...")
    data_processed, train, test = preprocess_data(data, symbols=SYMBOLS)
    data_processed.to_csv(PROCESSED_DATA_PATH, index=False)

    # 3. Calcular pesos óptimos
    #train
    expected_returns_train = train.groupby('symbol')['retorno_diario'].mean().values * 252
    returns_matrix_train = train.pivot(index='date', columns='symbol', values='retorno_diario').dropna().values * np.sqrt(252)
    cov_matrix_train = np.cov(returns_matrix_train, rowvar=False)
    MIN_RETURN_TRAIN, MAX_VOLATILITY_TRAIN = markowitz_parameters(expected_returns_train, cov_matrix_train, ALPHA_1, ALPHA_2, BETA_1, BETA_2)
    OPTIMAL_WEIGHTS_TRAIN = markowitz_weights(expected_returns_train, cov_matrix_train, MIN_RETURN_TRAIN, MAX_VOLATILITY_TRAIN)
    #todo
    expected_returns= data_processed.groupby('symbol')['retorno_diario'].mean().values * 252
    returns_matrix = data_processed.pivot(index='date', columns='symbol', values='retorno_diario').dropna().values * np.sqrt(252)
    cov_matrix= np.cov(returns_matrix, rowvar=False)
    MIN_RETURN, MAX_VOLATILITY = markowitz_parameters(expected_returns, cov_matrix, ALPHA_1, ALPHA_2, BETA_1, BETA_2)
    print("min_return", MIN_RETURN, "max_volat", MAX_VOLATILITY)
    OPTIMAL_WEIGHTS= markowitz_weights(expected_returns, cov_matrix, MIN_RETURN, MAX_VOLATILITY)
    print(OPTIMAL_WEIGHTS)

    # 4. Crear el entorno de entrenamiento
    env_train = Environment(train, optimal_weights=OPTIMAL_WEIGHTS_TRAIN, initial_sharpe_ratio=1.5, initial_capital=INITIAL_CAPITAL)

    # 5. Cargar o entrenar el modelo
    if os.path.exists(MODEL_PATH):
        print("Cargando modelo entrenado...")
        trained_model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Creando el modelo Q-Network...")
        input_dim = env_train.reset().shape[0]
        output_dim = 3 * env_train.num_assets
        hidden_size = 256
        q_model = Q_Network(input_dim, hidden_size, output_dim)
        q_model_target = Q_Network(input_dim, hidden_size, output_dim)

        print("Entrenando el modelo DQN...")
        trained_model, rewards = train_dqn(
            env_train, q_model, q_model_target,
            n_episodes=N_EPISODES,
            batch_size=BATCH_SIZE,
            num_assets=len(SYMBOLS)
        )

        print(f"Guardando el modelo en {MODEL_PATH}...")
        trained_model.save(MODEL_PATH, save_format="keras")

        print(f"Guardando recompensas del entrenamiento en {REWARDS_PATH}...")
        with open(REWARDS_PATH, 'w') as f:
            for reward in rewards:
                f.write(f"{reward}\n")


    # 6. Validar el modelo
    if not os.path.exists(VALIDATION_PATH):
        print("Iniciando validación del modelo con datos de prueba...")
        env_test = Environment(
            test, 
            optimal_weights=OPTIMAL_WEIGHTS_TRAIN, 
            initial_sharpe_ratio=1.5, 
            initial_capital=INITIAL_CAPITAL
        )
        compare_to_baseline(env_test, trained_model, save_path=VALIDATION_PATH)
        print_validation_data(VALIDATION_PATH)
    else:
        print("Modelo ya validado. Cargando e imprimiendo resultados de validación...")
        print_validation_data(VALIDATION_PATH)


    # 7. Generar recomendaciones
    print("Generando recomendaciones...")
    env = Environment(data_processed, optimal_weights=OPTIMAL_WEIGHTS, initial_sharpe_ratio=1.5, initial_capital=INITIAL_CAPITAL)
    recommendations_df = daily_recommendation(env, trained_model, INITIAL_CAPITAL)
    recommendations_df['MIN_RETURN'] = MIN_RETURN
    recommendations_df['MAX_VOLATILITY'] = MAX_VOLATILITY
    recommendations_df['OPTIMAL_WEIGHTS'] = [OPTIMAL_WEIGHTS] * len(recommendations_df)
    
    result_file = f"simulations/simulation{simulation_number}_result.csv"
    recommendations_df.to_csv(result_file, index=False)
    print(recommendations_df)
    print(f"Recomendaciones guardadas en: {result_file}")
 
    
if __name__ == "__main__":
    main()
