import matplotlib.pyplot as plt
import pandas as pd

def plot_rewards_episodes(train_rewardsA, train_rewardsB, epochs):
    """
    Genera una gráfica para comparar las recompensas acumuladas en el entrenamiento.

    Args:
        train_rewards (list): Lista de recompensas acumuladas por episodio durante el entrenamiento.
        title (str): Título de la gráfica.
    """
        # Convertir train_rewards a un array plano si es necesario

    plt.figure(figsize=(12, 6))
    
    # gráfica de recompensas de entrenamiento
    plt.plot(epochs, train_rewardsA, label="Recompensas de entrenamiento - Usuario A", color="mediumseagreen", alpha=0.7)
    plt.plot(epochs, train_rewardsB, label="Recompensas de entrenamiento - Usuario B", color="blue", alpha=0.7)
      
    # etiquetas y leyenda
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa acumulada")
    plt.legend()
    plt.grid(alpha=0.3)
  
    # mostrar la gráfica
    plt.show()

REWARDS_PATHA = f"model/saved_model/trained_model_rewards1.txt"
train_rewardsA = pd.read_csv(REWARDS_PATHA, header=None, names=["Reward"])

REWARDS_PATHB = f"model/saved_model/trained_model_rewards2.txt"
train_rewardsB = pd.read_csv(REWARDS_PATHB, header=None, names=["Reward"])
plot_rewards_episodes(train_rewardsA, train_rewardsB, epochs =  list(range(1, len(train_rewardsA) + 1)) ) 

