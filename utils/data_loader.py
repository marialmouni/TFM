import pandas as pd
import yfinance as yf

def load_data(symbols, start_date, save_path=None):
    """
    Carga datos de Yahoo Finance para una lista de símbolos.
    
    Args:
        symbols (list): Lista de símbolos de acciones.
        start_date (str): Fecha de inicio para los datos.
        save_path (str, optional): Ruta para guardar los datos cargados.

    Returns:
        pd.DataFrame: Datos combinados de todas las acciones.
    """
    data = pd.DataFrame()

    for symbol in symbols:
        temp_data = yf.download(symbol, start=start_date)
        # Eliminar el MultiIndex y renombrar las columnas para eliminar el ticker
        temp_data.columns = temp_data.columns.get_level_values(0)
        temp_data['symbol'] = symbol
        data = pd.concat([data, temp_data])

    data.reset_index(inplace=True)

    data.rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close',
        'Volume': 'volume'
    }, inplace=True)

    if save_path:
        data.to_csv(save_path, index=False)
    return data

