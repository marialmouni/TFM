import numpy as np
import random
import tensorflow as tf
from collections import deque

def train_dqn(env, Q, Q_ast, num_assets, n_episodes=1000, memory_size=10000, batch_size=64,
              gamma=0.99, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.997,
              train_freq=10, show_log_freq=10, callbacks = None):
    memory = deque(maxlen=memory_size)  # Memoria para almacenar las transiciones
    total_rewards = []  # Lista para almacenar las recompensas totales por episodio
    optimizer = tf.keras.optimizers.Adam()  # Optimizer para la red Q

    for epoch in range(n_episodes):
        obs = env.reset()  # Inicializa el estado del entorno
        total_reward = 0  # Inicializa la recompensa total para el episodio

        while True:
            # Selección de acción usando la política epsilon-greedy
            if np.random.rand() > epsilon:
                q_values = Q(tf.convert_to_tensor([obs], dtype=tf.float32))
                q_values = q_values.numpy()
                q_values = q_values.reshape(num_assets, 3)
                actions = np.argmax(q_values, axis=1)  # mejor acción para cada activo
            else:
                actions = np.random.randint(3, size=env.num_assets)  # Acción aleatoria

            # Realiza la acción y obtiene el siguiente estado y recompensa
            next_obs, rewards, done = env.step(actions)

            # Almacena la transición en la memoria
            memory.append((obs, actions, rewards, next_obs, done))

            # Actualiza el estado actual
            obs = next_obs
            total_reward += sum(rewards)

            if done:
                break

        # Entrenamiento de la red Q cada ciertos episodios
        if epoch % train_freq == 0 and len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            # Prepara los arreglos para las observaciones, acciones, recompensas, etc.
            b_pobs = np.array([entry[0] for entry in batch], dtype=np.float32)  # Observaciones previas
            b_pact = np.array([entry[1] for entry in batch])  # Acciones tomadas
            b_reward = np.array([entry[2] for entry in batch], dtype=np.float32)  # Recompensas obtenidas
            b_obs = np.array([entry[3] for entry in batch], dtype=np.float32)  # Observaciones posteriores
            b_done = np.array([entry[4] for entry in batch], dtype=bool)  # Estado de finalización

            # Asegúrate de que b_pact y b_reward tienen las dimensiones correctas
            if len(b_pact.shape) == 1:
                b_pact = np.expand_dims(b_pact, axis=-1)  # Expande dimensiones para un solo activo
                b_reward = np.expand_dims(b_reward, axis=-1)  # Expande dimensiones para un solo activo

            # Asegúrate de que b_pact y b_reward tienen las dimensiones correctas
            assert b_pact.shape == (batch_size, env.num_assets)
            assert b_reward.shape == (batch_size, env.num_assets)

            # Cálculo de la pérdida y actualización de pesos
            with tf.GradientTape() as tape:
                q_values = Q(tf.convert_to_tensor(b_pobs, dtype=tf.float32))  # Predicción de Q para el batch
                maxq = np.max(Q_ast(tf.convert_to_tensor(b_obs, dtype=tf.float32)), axis=1)  # Q máxima para el siguiente estado

                # Copiar las predicciones de Q
                target = np.copy(q_values.numpy())  # Copia de las predicciones de Q para actualizar
                q_indices = np.arange(batch_size)  # Índices del batch

                # Actualización del target con la fórmula de DQN
                for k in range(env.num_assets):
                    action_indices = b_pact[:, k] + k * 3  # Índice para cada acción por activo
                    target[q_indices, action_indices] = b_reward[:, k] + gamma * maxq * (~b_done)

                # Calcula la pérdida de la función de valor (error cuadrático medio)
                loss = tf.reduce_mean(tf.square(q_values - target))  # Error cuadrático medio
            grads = tape.gradient(loss, Q.trainable_variables)  # Gradientes
            optimizer.apply_gradients(zip(grads, Q.trainable_variables))  # Actualiza los pesos de Q

        # Llamar a los callbacks al final de cada epoch
        if callbacks:
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={"reward": total_reward})

        # Actualiza los pesos de la red objetivo (Q_ast) cada ciertos episodios
        if epoch % train_freq == 0:
            Q_ast.set_weights(Q.get_weights())

        # Disminuye el valor de epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        total_rewards.append(total_reward)  # Almacena la recompensa total para este episodio

        # Imprime el progreso cada ciertos episodios
        if (epoch + 1) % show_log_freq == 0:
            print(f"Epoch {epoch + 1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return Q, total_rewards  # Devuelve la red Q entrenada y las recompensas totales
