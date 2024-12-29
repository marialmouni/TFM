import numpy as np

class Environment:
    """
    Clase que modela un entorno de mercado financiero para la simulación de estrategias de trading.

    Args:
        data (DataFrame): Datos históricos del mercado, con columnas 'symbol' y 'close'.
        optimal_weights (dict o ndarray): Pesos óptimos iniciales asignados a cada activo.
        initial_capital (float): Capital inicial disponible. Por defecto es 100.
        history_t (int): Longitud del historial para las observaciones. Por defecto es 90.

    Métodos:
        reset(): Reinicia el entorno al estado inicial.
        _get_observation(): Genera las observaciones actuales del entorno.
        _calculate_sharpe(symbol): Calcula el ratio de Sharpe para un símbolo dado.
        step(act): Ejecuta un paso en el entorno con las acciones proporcionadas.
    """

    def __init__(self, data, optimal_weights, initial_sharpe_ratio, initial_capital=100, history_t=90):
        """
        Inicializa el entorno con datos históricos, pesos óptimos y configuración inicial.

        Args:
            data (DataFrame): Datos históricos con información de precios de cierre y símbolos.
            optimal_weights (dict o ndarray): Pesos iniciales para cada símbolo.
            initial_capital (float): Capital inicial disponible para operar.
            history_t (int): Número de pasos históricos considerados en las observaciones.
        """

        self.data = data
        self.history_t = history_t
        self.symbols = list(data['symbol'].unique()) # lista de activos
        self.num_assets = len(self.symbols)
        self.optimal_weights = optimal_weights  # pesos iniciales optimizados
        self.initial_sharpe_ratio = initial_sharpe_ratio  # Sharpe Ratio inicial
        self.initial_capital = initial_capital  # capital inicial
        self.best_fraction = 1.0  # Valor predeterminado (100%)

        # convertir optimal_weights a diccionario si es un ndarray
        if isinstance(optimal_weights, (np.ndarray, list)):
            self.optimal_weights = dict(zip(self.symbols, optimal_weights))
        else:
            self.optimal_weights = optimal_weights

        # calcular log-retornos y volatilidad histórica

        # Calcular log-retornos de forma vectorizada
        self.data['log_return'] = np.log(self.data.groupby('symbol')['close'].pct_change() + 1)

        # Calcular volatilidad histórica con rolling
        self.data['volatility'] = self.data.groupby('symbol')['log_return'].rolling(window=20).std().reset_index(level=0, drop=True)

        # Rellenar valores NaN con 0 (por ejemplo, al principio de las series)
        self.data['log_return'] = self.data['log_return'].fillna(0)
        self.data['volatility'] = self.data['volatility'].fillna(0)


        # definir espacios de observación y acción
        self.observation_space = np.zeros(self.num_assets + self.num_assets * (1 + self.history_t))
        self.action_space = np.arange(3)  # Acciones: 0 (stay), 1 (buy), 2 (sell)

        self.reset()


    def reset(self):
        """
        Reinicia el entorno al estado inicial.
        Configura el historial, posiciones iniciales y Sharpe Ratio para cada activo.

        Returns:
            np.ndarray: Observaciones iniciales del entorno.
        """

        self.t = 0
        self.done = False
        self.profits = {symbol: 0 for symbol in self.symbols}
        self.positions = {}
        self.position_value = {}
        self.history = {symbol: [0 for _ in range(self.history_t)] for symbol in self.symbols}
        # Almacenar la mejor fracción antes de reiniciar
        if hasattr(self, 'best_fraction'):
            self.previous_best_fraction = self.best_fraction
        else:
            self.previous_best_fraction = 1  # Valor por defecto si no existe

        # configurar posiciones iniciales según los pesos
        for symbol in self.symbols:
            price = self.data.loc[self.data['symbol'] == symbol, 'close'].iloc[self.t]
            weight = self.optimal_weights.get(symbol, 0)
            self.positions[symbol] = [self.initial_capital * weight / price]  # cantidad comprada
            self.position_value[symbol] = self.positions[symbol][0] * price  # valor inicial

        # guardar el último Sharpe Ratio para comparar
        self.last_sharpe = {symbol: self.initial_sharpe_ratio for symbol in self.symbols}

        return self._get_observation()


    def _get_observation(self):
        """
        Genera las observaciones actuales del entorno.

        Returns:
            np.ndarray: Vector de observaciones compuesto por valores de posición, volatilidad
                        y el historial de retornos.
        """

        obs = []
        for symbol in self.symbols:
            # Volatilidad actual
            if self.t >= len(self.data.loc[self.data['symbol'] == symbol]):
                volatility = 0  # Valor por defecto si el índice es inválido
            else:
                volatility = self.data.loc[self.data['symbol'] == symbol, 'volatility'].iloc[self.t]

            # Valores no normalizados
            position_value = self.position_value[symbol]
            history_returns = self.history[symbol][-self.history_t:]

            # Normalización Min-Max (ejemplo)
            position_value = position_value / self.initial_capital  # Escalar entre 0 y 1 usando el capital inicial
            volatility = volatility / (self.data['volatility'].max() + 1e-8)  # Evitar división por cero
            history_returns = np.array(history_returns) / (np.max(np.abs(history_returns)) + 1e-8)

            # Concatenar características normalizadas
            obs.append(position_value)
            obs.append(volatility)
            obs.extend(history_returns)
        return np.array(obs, dtype=np.float32)



    def _calculate_sharpe(self, symbol):
        """
        Calcula el ratio de Sharpe para un activo basado en su historial.

        Args:
            symbol (str): Símbolo del activo.

        Returns:
            float: Ratio de Sharpe calculado.
        """

        if len(self.history[symbol]) > 1:

            periods_per_year = 252 # retornos diarios

            avg_return = np.mean(self.history[symbol])
            std_dev = np.std(self.history[symbol])
            avg_return *= periods_per_year  # Donde periods_per_year es 252 para retornos diarios
            std_dev *= np.sqrt(periods_per_year)
            risk_free_rate = 0.01  # Tasa libre de riesgo anualizada (ejemplo: 1%)
            sharpe_ratio = (avg_return - (risk_free_rate / periods_per_year)) / (std_dev + 1e-8)

            return sharpe_ratio  # evitar división por cero

        return 0

    def step(self, act):
        """
        Ejecuta un paso en el entorno basado en las acciones dadas, probando diferentes fracciones para la compra y la venta.

        Args:
            act (list): Lista de acciones para cada activo (0: mantener, 1: comprar, 2: vender).
            fractions (list): Lista de fracciones a probar para la compra/venta (por ejemplo, [0.1, 0.2, 0.5]).

        Returns:
            tuple:
                - np.ndarray: Nueva observación después de realizar las acciones.
                - list: Recompensas obtenidas para cada activo.
                - bool: Indicador de si el episodio ha terminado.
        """

        best_reward = -float('inf')  # Inicializa la mejor recompensa como negativa infinita
        best_fraction = None  # Fracción que genera la mejor recompensa total
        best_reward_per_symbol = [0] * self.num_assets  # Recompensas por símbolo
        fractions = [i/10 for i in range(1, 11)]  # se prueban fracciones de 0.1 a 1.0

        # Se probarán todas las fracciones
        for fraction in fractions:
            reward = [0] * self.num_assets  # Inicializa el vector de recompensas para esta fracción

            for idx, symbol in enumerate(self.symbols):
                if idx >= len(act):  # Verificar que idx sea menor que el tamaño de act
                    break

                # Filtrar datos de precios para el símbolo
                filtered_data = self.data.loc[self.data['symbol'] == symbol, 'close']
                if self.t >= len(filtered_data):
                    self.done = True
                    break  # Salir del bucle si hemos terminado los datos para cualquier símbolo

                price = self.data.loc[self.data['symbol'] == symbol, 'close'].iloc[self.t]  # Precio actual del activo

                # act = 0: stay, 1: buy, 2: sell
                if act[idx] == 1:  # Buy (comprar una fracción)
                    # Cálculo de la cantidad a comprar
                    max_amount_to_buy = self.initial_capital * self.optimal_weights.get(symbol, 0) / price
                    amount_to_buy = max_amount_to_buy * fraction  # Solo compra el porcentaje de la cantidad máxima
                    self.positions[symbol].append(price)  # Añadir el precio de compra
                    reward[idx] += amount_to_buy * price  # Recompensa por comprar una fracción

                elif act[idx] == 2:  # Sell (vender una fracción)
                    if len(self.positions[symbol]) == 0:  # No hay posiciones que vender
                        reward[idx] -= price * 0.1  # Penalización por intentar vender sin tener posiciones
                    else:
                        amount_to_sell = len(self.positions[symbol]) * fraction  # Vender solo una fracción
                        amount_to_sell = min(amount_to_sell, len(self.positions[symbol]))  # No vender más de lo que se tiene
                        # Calcular las ganancias
                        profits = 0
                        for _ in range(int(amount_to_sell)):
                            buy_price = self.positions[symbol].pop(0)  # Vender las primeras posiciones
                            profits += price - buy_price
                        reward[idx] += profits  # Recompensa por la venta de una fracción
                        self.profits[symbol] += profits  # Actualizar las ganancias del símbolo

            # Calcular la recompensa total de este paso para esta fracción
            total_reward = sum(reward)

            # Si esta fracción genera una mejor recompensa, la seleccionamos
            if total_reward > best_reward:
                best_reward = total_reward
                best_fraction = fraction  # Guardar la fracción que generó la mejor recompensa

            best_reward_per_symbol = reward  # Actualizar la recompensa por activo

        # Penalización por no mejorar el Sharpe Ratio
        for idx, symbol in enumerate(self.symbols):
            current_sharpe = self._calculate_sharpe(symbol)
            if current_sharpe <= self.last_sharpe[symbol]:  # Si el Sharpe Ratio no mejora
                position_value = self.position_value[symbol]  # Valor de la posición actual
                penalty = position_value * 0.1  # Penalización proporcional al valor de la posición
                best_reward_per_symbol[idx] -= penalty
            else:
                self.last_sharpe[symbol] = current_sharpe  # Actualizar el último Sharpe Ratio

        # Actualizar `best_fraction` en el entorno
        self.best_fraction = best_fraction if best_fraction is not None else 1.0
        
        # Clipping de recompensas para estabilidad
        reward = [np.tanh(r) for r in best_reward_per_symbol]

        # Avance de tiempo
        self.t += 1

        # Generar nueva observación
        next_obs = self._get_observation()

        return next_obs, reward, self.done  # Solo devolvemos lo esencial: estado siguiente, recompensas y si terminó el episodio