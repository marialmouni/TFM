import pandas as pd

def preprocess_data(data, symbols):
    """
    Procesa los datos para sincronizar fechas y calcular retornos.

    Args:
        data (pd.DataFrame): Datos crudos.

    Returns:
        pd.DataFrame: Datos procesados.
    """
    # Filtrar fechas comunes
    common_dates = data.groupby('symbol')['date'].apply(set).agg(set.intersection)
    common_dates = set.intersection(*common_dates)

    # Filtrar el DataFrame
    data_synchronized = data[data['date'].isin(common_dates)].copy()
    data_synchronized.dropna(inplace=True)

    # Calcular retornos diarios
    data_synchronized['retorno_diario'] = (
        data_synchronized.groupby('symbol', group_keys=False)['close']
    .apply(lambda x: x.pct_change())
    ).reindex(data_synchronized.index).values

    # Ordenar por fecha
    data_synchronized = data_synchronized.sort_values('date')
    
    # Resetear el índice para evitar índices duplicados
    data_synchronized.reset_index(drop=True, inplace=True)

    # Asegurarse de que la columna 'date' esté en formato datetime
    data_synchronized['date'] = pd.to_datetime(data_synchronized['date'])

    # Preparar subconjuntos train(entrenamiento) y test(validación)
    grouped_data = data_synchronized.groupby('symbol')

    train = pd.DataFrame()
    test = pd.DataFrame()

    # Calcula el índice de corte para el 80% de entrenamiento y 20% de prueba
    train_size = int((len(data) * 0.8)/(len(symbols)))
    for symbol, group in grouped_data:
    # Divide el grupo en entrenamiento y prueba
        train_group = group.iloc[:train_size]
        test_group = group.iloc[train_size:]
        # Concatenar los grupos correspondientes a entrenamiento y prueba
        train = pd.concat([train, train_group])
        test = pd.concat([test, test_group])

    train.reset_index(drop=True, inplace=True)

    return data_synchronized, train, test
