import numpy as np
import tensorflow as tf

def validate_dqn(env, Q):
    obs = env.reset()  # Resetear el entorno
    total_reward = 0

    while not env.done:
        # Selección de acción usando la política greedy
        q_values = Q(tf.convert_to_tensor([obs], dtype=tf.float32))
        q_values = q_values.numpy()
        q_values = q_values.reshape(env.num_assets, 3)
        actions = np.argmax(q_values, axis=1)  # Mejor acción para cada activo

        # Realiza la acción y obtiene el siguiente estado y recompensa
        next_obs, rewards, done = env.step(actions)
        obs = next_obs
        total_reward += sum(rewards)

    return total_reward

