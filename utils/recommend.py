import tensorflow as tf
import numpy as np
import pandas as pd

def generate_daily_recommendations(env, actions, initial_capital):
    recommendations = {}
    for idx, symbol in enumerate(env.symbols):
        current_price = env.data.loc[(env.data['symbol'] == symbol), 'close'].iloc[env.t]

        if actions[idx] == 0:  # Mantener
            recommendations[symbol] = "Mantener"
        elif actions[idx] == 1:  # Comprar
            max_amount_to_buy = initial_capital * env.optimal_weights.get(symbol, 0) / current_price
            amount_to_buy = max_amount_to_buy * env.best_fraction  # Usar la mejor fracción calculada
            recommendations[symbol] = f"Comprar {amount_to_buy:.2f} unidades"
        elif actions[idx] == 2:  # Vender
            amount_to_sell = len(env.positions[symbol]) * env.best_fraction
            recommendations[symbol] = f"Vender {amount_to_sell:.2f} unidades"
    return recommendations

def daily_recommendation(env, trained_model, INITIAL_CAPITAL):
    # Reinicia el entorno y obtén el estado inicial
    obs = env.reset()  # Estado inicial basado en los datos históricos
    # Predecir valores Q para el estado actual
    q_values = trained_model(tf.convert_to_tensor([obs], dtype=tf.float32))
    q_values = q_values.numpy().reshape(env.num_assets, -1)  # Formato: (num_assets, num_actions)

    # Seleccionar la mejor acción para cada activo
    actions = np.argmax(q_values, axis=1)

    recommendations = generate_daily_recommendations(env, actions, initial_capital=INITIAL_CAPITAL)

    # Convertir las recomendaciones a un DataFrame
    recommendations_list = [{"Activo": symbol, "Recomendación": action} for symbol, action in recommendations.items()]
    recommendations_df = pd.DataFrame(recommendations_list)

    return recommendations_df