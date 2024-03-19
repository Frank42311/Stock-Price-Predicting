import pandas as pd
import numpy as np
import talib
import os
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.decomposition import PCA

# Paths for stock data and indicators
stocks_path = r'data\\us_stocks\\'
stocks_indicator_path = r'data\\us_stocks_indicators\\'

# -------------------------------------------------------------
# Creation and saving of the date scaler
# Define your minimum and maximum dates
min_date = pd.Timestamp('1990-03-26')
max_date = pd.Timestamp('2023-03-15')

# Create a scaler for date values
datetime_scaler = MinMaxScaler()

# Fit the scaler using the Unix timestamps of the min and max dates
datetime_scaler.fit([[min_date.timestamp()], [max_date.timestamp()]])

# Save the scaler to a file for later use
save_path_datetime = "data\\scalers\\datetime_scalers"
joblib.dump(datetime_scaler, os.path.join(save_path_datetime, "datetime_scaler.save"))

def process_single_stock(stock_name: str) -> None:
    """
    Processes a single stock file, calculating various technical indicators and normalizing certain values.
    
    Parameters:
    - stock_name (str): The name of the stock file to be processed.
    
    Returns:
    - None: This function directly modifies the DataFrame but doesn't return any value.
    """

    # Construct the file path for the stock
    single_stock = stocks_path + stock_name

    # Read stock data from the file
    df = pd.read_csv(single_stock)

    # Rename columns to more understandable names
    df.columns = ['code', 'date', 'open_post', 'close_post', 'high_post', 'low_post', 'volume', 'amount',
                  'volatility', 'change_percent', 'change_amount', 'turnover', 'open', 'close',
                  'high', 'low']

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    # Select relevant columns for further processing
    df = df[['date', 'change_percent', 'change_amount', 'volume', 'open', 'close', 'high', 'low']]

    #################################################################################################################################
    # 
    # Calculation of Technical Indicators
    # 
    #################################################################################################################################

    ''' Accumulation Distribution Index (ADI) '''
    # 20-day Moving Average
    df['20MA'] = talib.MA(df['close'], timeperiod=20)
    # Calculate the money flow multiplier
    high_low = df['high'] - df['low']
    money_flow_multiplier = np.where(high_low == 0, 0,
                                     ((2 * df['close'] - df['high'] - df['low']) * (df['volume'] / 1e6)) / high_low)
    # Calculate the money flow volume
    money_flow_volume = money_flow_multiplier * df['volume']
    # Cumulative sum of money flow volume gives ADI
    df['ADI'] = money_flow_volume.cumsum()

    ''' Average Directional Index (ADX) '''
    df["ADX"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

    ''' Bollinger Bands (BB) - Using 20-day MA and 2 standard deviations '''
    std = df['close'].rolling(20).std()
    df['BB_Upper'] = df['20MA'] + 2 * std
    df['BB_Lower'] = df['20MA'] - 2 * std

    ''' Commodity Channel Index (CCI) '''
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)

    ''' Double Exponential Moving Average (DEMA) for 10 and 20-day periods '''
    df['10EMA'] = df['close'].ewm(span=10, adjust=False).mean()
    df['10DEMA'] = 2 * df['10EMA'] - df['10EMA'].ewm(span=10, adjust=False).mean()
    df['20EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['20DEMA'] = 2 * df['20EMA'] - df['20EMA'].ewm(span=20, adjust=False).mean()

    ''' Detrended Price Oscillator (DPO) '''
    df['10MA'] = talib.MA(df['close'], timeperiod=10)
    df['DPO'] = df['close'].shift(10) - df['10MA']

    ''' Exponential Moving Averages (EMA) for 5, 10, and 30-day periods '''
    df['30EMA'] = df['close'].ewm(span=30, adjust=False).mean()
    df['5EMA'] = talib.EMA(df['close'], timeperiod=5)

    ''' Ease of Movement (EMV) '''
    df['EMV'] = ((df['high'] + df['low']) / 2) * ((df['high'] - df['low']) / df['volume']) * (
            (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2)
    df['EMV_MA'] = df['EMV'].rolling(9).mean()

    ''' Keltner Channel (KC) '''
    df['TR'] = talib.TRANGE(df['high'], df['low'], df['close'])
    df['ATR20'] = df['TR'].rolling(20).mean()
    df['KC_Upper'] = df['20MA'] + 2 * df['ATR20']
    df['KC_Lower'] = df['20MA'] - 2 * df['ATR20']

    ''' KDJ Indicator with periods of 9 and 3 '''
    # Calculate the highest and lowest prices over the past N days
    high_n = df['high'].rolling(window=9, min_periods=0).max()
    low_n = df['low'].rolling(window=9, min_periods=0).min()
    # Compute the RSV (Raw Stochastic Value)
    df['RSV'] = (df['close'] - low_n) / (high_n - low_n) * 100
    # Initialize K and D with zeros
    df['K'] = np.zeros(len(df))
    df['D'] = np.zeros(len(df))
    # Set the initial values of K and D to 50
    df.loc[0, 'K'] = 50
    df.loc[0, 'D'] = 50
    # Calculate the KDJ indicator
    for i in range(1, len(df)):
        df.loc[i, 'K'] = 2 / 3 * df.loc[i - 1, 'K'] + 1 / 3 * df.loc[i, 'RSV']
        df.loc[i, 'D'] = 2 / 3 * df.loc[i - 1, 'D'] + 1 / 3 * df.loc[i, 'K']
    # Calculate the J line as 3*D - 2*K
    df['J'] = 3 * df['D'] - 2 * df['K']

    ''' Moving Averages (MA) '''
    # Calculate 10-day and 30-day moving averages
    df['30MA'] = talib.MA(df['close'], timeperiod=30)
    df['5MA'] = talib.MA(df['close'], timeperiod=5)

    ''' Moving Average Convergence Divergence (MACD) with periods of 12 and 26 '''
    # The MACD is a trend-following momentum indicator
    df['DIF'], df['DEA'], macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    ''' Moving Average Envelope (MAE) '''
    # Set envelope percentage
    k = 3
    # Calculate the upper and lower envelopes
    df['MAE_Upper'] = df['20MA'] * (1 + k / 100)
    df['MAE_Lower'] = df['20MA'] * (1 - k / 100)

    ''' Money Flow Index (MFI) with a period of 14 '''
    # The MFI is a volume-weighted RSI
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    # Calculate the signal line as a simple moving average of MFI
    df['MFI_MA'] = talib.SMA(df['MFI'], 9)

    ''' Momentum (MOM) '''
    # The MOM indicator measures the rate of change
    df['MOM'] = talib.MOM(df['close'], timeperiod=14)

    ''' On-Balance Volume (OBV) '''
    # The OBV measures buying and selling pressure
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    ''' Oscillator (OSC) '''
    # The OSC is the difference between two exponential moving averages
    ema12 = talib.EMA(df['close'], timeperiod=12)
    ema26 = talib.EMA(df['close'], timeperiod=26)
    df['OSC'] = ema12 - ema26

    ''' Parabolic SAR (PSAR) '''
    # The PSAR indicator is used to determine the direction of a stock's momentum
    df['PSAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)

    ''' Relative Exponential Index (REX) '''
    # The REX compares short-term and long-term exponential moving averages
    short_ema = talib.EMA(df['close'], timeperiod=2)
    df['REX'] = (short_ema - df['10EMA']) / df['10EMA']

    ''' Weighted Moving Average (RMA) '''
    # Calculate 10-day and 30-day weighted moving averages
    df['RMA10'] = talib.WMA(df['close'], timeperiod=10)
    df['RMA30'] = talib.WMA(df['close'], timeperiod=30)

    ''' Relative Strength Index (RSI) '''
    # The RSI is a momentum oscillator measuring the speed and change of price movements
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)

    ''' Triple Exponential Moving Average (TEMA) '''
    # The TEMA reduces lag by applying multiple EMA calculations to the price
    df['10TEMA'] = talib.TEMA(df['close'], timeperiod=10)
    df['20TEMA'] = talib.TEMA(df['close'], timeperiod=20)
    df['30TEMA'] = talib.TEMA(df['close'], timeperiod=30)

    ''' TRIX '''
    # The TRIX is a momentum oscillator that oscillates around zero
    df['TRIX12'] = talib.TRIX(df['close'], timeperiod=12)
    df['TRIX24'] = talib.TRIX(df['close'], timeperiod=24)

    ''' Ultimate Oscillator (UO) '''
    # The UO combines short, medium, and long-term price action into one oscillator
    df['UO'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['UO_fast_ma'] = df['UO'].rolling(window=3).mean()
    df['UO_slow_ma'] = df['UO'].rolling(window=7).mean()

    ''' Vortex Indicator (VI) '''
    # The VI identifies the start of a new trend or the continuation of an existing trend
    atr = talib.SMA(df['TR'], timeperiod=14)
    pdm = talib.PLUS_DM(df['high'], df['low'])
    ndm = talib.MINUS_DM(df['high'], df['low'])
    pvm = pdm / atr
    nvm = ndm / atr
    df['vi_plus'] = talib.SMA(pvm, timeperiod=14)
    df['vi_minus'] = talib.SMA(nvm, timeperiod=14)
    df['VI'] = abs(df['vi_plus'] - df['vi_minus']) / (df['vi_plus'] + df['vi_minus'])

    ''' Williams %R (W%R) '''
    # The W%R is a momentum indicator that measures overbought and oversold levels
    df['WR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

    df = df.copy()

    #################################################################################################################################
    # 
    # cdl patterns
    # 
    #################################################################################################################################

    df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)

    # Overlap studies
    df['he_trendline'] = talib.HT_TRENDLINE(df['Close'])
    df['kama'] = talib.KAMA(df['Close'], timeperiod=30)
    df['mama'], df['fama'] = talib.MAMA(df['Close'])
    df['midpoint'] = talib.MIDPOINT(df['Close'], timeperiod=14)
    df['midprice'] = talib.MIDPRICE(df['High'], df['Low'], timeperiod=14)
    df['t3'] = talib.T3(df['Close'], timeperiod=5, vfactor=0)
    df['trima'] = talib.TRIMA(df['Close'], timeperiod=30)
    df['wma'] = talib.WMA(df['Close'], timeperiod=30)

    # Momentum
    df['adxr'] = talib.ADXR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['aroondown'], df['aroonup'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
    df['aroonose'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
    df['bop'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])
    df['cmo'] = talib.CMO(df['Close'], timeperiod=14)
    df['dx'] = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['macdext'], df['macdsignalext'], df['macdhistext'] = talib.MACDEXT(df['Close'], fastperiod=12, fastmatype=0,
                                                                    slowperiod=26, slowmatype=0,
                                                                    signalperiod=9, signalmatype=0)
    df['macdfix'], df['macdsignalfix'], df['macdhistfix'] = talib.MACDFIX(df['Close'], signalperiod=9)
    df['minus_di'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['minus_dm'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    df['plus_di'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['plus_dm'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    df['ppo'] = talib.PPO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
    df['roc'] = talib.ROC(df['Close'], timeperiod=10)
    df['rocp'] = talib.ROCP(df['Close'], timeperiod=10)
    df['rocr'] = talib.ROCR(df['Close'], timeperiod=10)
    df['rocr100'] = talib.ROCR100(df['Close'], timeperiod=10)
    df['slowk'], df['slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0,
                                     slowd_period=3,
                                     slowd_matype=0)
    df['fastk'], df['fastd'] = talib.STOCHF(df['High'], df['Low'], df['Close'], fastk_period=5, fastd_period=3,
                                      fastd_matype=0)
    df['fastk'], df['fastd'] = talib.STOCHRSI(df['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['ultosc'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['willr'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Volume
    df['ad'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    df['adosc'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
    df['obv'] = talib.OBV(df['Close'], df['Volume'])

    # Volatility
    df['matr'] = talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['trange'] = talib.TRANGE(df['High'], df['Low'], df['Close'])

    df = df.copy()

    # Pattern Recognition
    # Buillish
    df['cdl3whitesoldiers'] = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlmarubozu'] = talib.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdltakuri'] = talib.CDLTAKURI(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlmorningstar'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)
    df['cdlladderbottom'] = talib.CDLLADDERBOTTOM(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdl3starsinsouth'] = talib.CDL3STARSINSOUTH(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdl3inside'] = talib.CDL3INSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlconcealbabyswall'] = talib.CDLCONCEALBABYSWALL(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdl3outside'] = talib.CDL3OUTSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlseparatinglines'] = talib.CDLSEPARATINGLINES(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdltasukigap'] = talib.CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlunique3river'] = talib.CDLUNIQUE3RIVER(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlkicking'] = talib.CDLKICKING(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlkickingbylength'] = talib.CDLKICKINGBYLENGTH(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlengulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlpiercing'] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])

    # Condition Buillish
    df['cdlabandonedbaby'] = talib.CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)
    df['cdldoji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdldragonflydoji'] = talib.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlhammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlinvertedhammer'] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdllongleggeddoji'] = talib.CDLLONGLEGGEDDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlmorningdojistar'] = talib.CDLMORNINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)
    df['cdlhomingpigeon'] = talib.CDLHOMINGPIGEON(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlharamicross'] = talib.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlmathold'] = talib.CDLMATHOLD(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)

    # Bearish
    df['cdl3blackcrows'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdldarkcloudcover'] = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)
    df['cdlgravestonedoji'] = talib.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlshootingstar'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlclosingmarubozu'] = talib.CDLCLOSINGMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdleveningstar'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)
    df['cdl2crows'] = talib.CDL2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlidentical3crows'] = talib.CDLIDENTICAL3CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdladvanceblock'] = talib.CDLADVANCEBLOCK(df['Open'], df['High'], df['Low'], df['Close'])

    # Condition Bearish
    df['cdlbelthold'] = talib.CDLBELTHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlbreakaway'] = talib.CDLBREAKAWAY(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdleveningdojistar'] = talib.CDLEVENINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)
    df['cdlhangingman'] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlcounterattack'] = talib.CDLCOUNTERATTACK(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlonneck'] = talib.CDLONNECK(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlstalledpattern'] = talib.CDLSTALLEDPATTERN(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlupsidegap2crows'] = talib.CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Close'])

    # Reverse
    df['cdldojistar'] = talib.CDLDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlharami'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlhighwave'] = talib.CDLHIGHWAVE(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlhikkake'] = talib.CDLHIKKAKE(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlhikkakemod'] = talib.CDLHIKKAKEMOD(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlinneck'] = talib.CDLINNECK(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlrickshawman'] = talib.CDLRICKSHAWMAN(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlrisefall3methods'] = talib.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlshortline'] = talib.CDLSHORTLINE(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlspinningtop'] = talib.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdltristar'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlthrusting'] = talib.CDLTHRUSTING(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdl3linestrike'] = talib.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])
    df['cdlgapsidesidewhite'] = talib.CDLGAPSIDESIDEWHITE(df['Open'], df['High'], df['Low'], df['Close'])

    #################################################################################################################################
    # 
    # Date Processing
    # 
    #################################################################################################################################

    # Convert the 'date' column to Unix timestamp format
    df['date_timestamp'] = df['date'].apply(lambda x: x.timestamp())

    # Ensure conversion to a 2D array. reshape(-1, 1) ensures it has the correct shape for scaling.
    dates_as_timestamps = df['date_timestamp'].values.reshape(-1, 1)

    # Now, we can normalize these 2D array dates
    normalized_dates = datetime_scaler.transform(dates_as_timestamps)

    # Reassign the normalized dates back to the 'date' column
    df['date'] = normalized_dates

    # Optionally, drop the timestamp column as it's no longer needed
    df.drop(columns=['date_timestamp'], inplace=True)

    #################################################################################################################################
    # 
    # OHLC and Volume Data Processing
    # 
    #################################################################################################################################

    def normalize_and_save_scalers(df, save_path_price_vol, stock_name):
        """
        Normalizes the price and volume columns of the dataframe, and saves the respective scalers.
        
        Parameters:
        - df (DataFrame): The dataframe containing stock data.
        - save_path_price_vol (str): The path where the scalers will be saved.
        - stock_name (str): The name of the stock, used for naming the saved scaler files.
        
        Returns:
        - None: This function modifies the dataframe in-place and saves scalers to files.
        """
        # Normalize price and volume
        price_scaler = MinMaxScaler()
        volume_scaler = MinMaxScaler()

        # Fit and transform the price columns
        df[['open', 'low', 'high', 'close']] = price_scaler.fit_transform(df[['open', 'low', 'high', 'close']])
        # Fit and transform the volume column
        df['volume'] = volume_scaler.fit_transform(df['volume'].values.reshape(-1, 1))

        # Ensure the save path exists
        os.makedirs(save_path_price_vol, exist_ok=True)

        # Save the scalers for price and volume
        joblib.dump(price_scaler, os.path.join(save_path_price_vol, f"{stock_name}_price.pkl"))
        joblib.dump(volume_scaler, os.path.join(save_path_price_vol, f"{stock_name}_volume.pkl"))

    # Path where the price and volume scalers will be saved
    save_path_price_vol = "data\\scalers\\price_vol_scalers"
    # Call the function to normalize and save scalers
    normalize_and_save_scalers(df, save_path_price_vol, stock_name)

    #################################################################################################################################
    # 
    # Indicator Normalization and PCA Transformation
    # cdl patterns Integration
    # 
    # Integrate OHCL, date, Indicators, cdl patterns
    # 
    #################################################################################################################################

    #################################################################################################################################
    # 
    # Indicator Normalization and PCA Transformation
    # 
    #################################################################################################################################


    if df.shape[0] > 180:
        # Dropping the first 90 rows
        df = df.iloc[90:].reset_index(drop=True)

        # Check for NaN values in the DataFrame
        nan_count = df.isna().sum().sum()
        if nan_count == 0:
            # List of technical indicators to process
            indicator_list = [
                "20MA", "ADI", "ADX", "BB_Upper", "BB_Lower", "CCI", "10EMA", "10DEMA",
                "20EMA", "20DEMA", "10MA", "DPO", "30EMA", "5EMA", "EMV", "EMV_MA", "TR",
                "ATR20", "KC_Upper", "KC_Lower", "RSV", "K", "D", "J", "30MA", "5MA", "DIF",
                "DEA", "MAE_Upper", "MAE_Lower", "MFI", "MFI_MA", "MOM", "OBV", "OSC", "PSAR",
                "REX", "RMA10", "RMA30", "RSI", "10TEMA", "20TEMA", "30TEMA", "TRIX12",
                "TRIX24", "UO", "UO_fast_ma", "UO_slow_ma", "vi_plus", "vi_minus", "VI", "WR",
                "he_trendline", "kama", "mama", "fama", "midpoint", "midprice", "t3", "trima",
                "wma", "adxr", "aroondown", "aroonup", "aroonose", "bop", "cmo", "dx", "macdext",
                "macdsignalext", "macdhistext", "macdfix", "macdsignalfix", "macdhistfix",
                "minus_di", "minus_dm", "plus_di", "plus_dm",
                "ppo", "roc", "rocp", "rocr", "rocr100", "slowk", "slowd", "fastk", "fastd",
                "ultosc", "willr", "ad", "adosc", "obv", "matr", "trange"
            ]
            
            # Extracting the indicators from the DataFrame
            indicator_df = df[indicator_list]

            # Normalize the indicator data
            scaler = MinMaxScaler()
            scaled_indicator_data = scaler.fit_transform(indicator_df)

            # Convert the normalized data back into a DataFrame
            scaled_indicator_df = pd.DataFrame(scaled_indicator_data, columns=indicator_list)

            # Apply PCA transformation
            pca = PCA(n_components=8)  # We choose 8 principal components
            principalComponents = pca.fit_transform(scaled_indicator_df)

            # Convert principal components into a DataFrame
            columns = ['indicator_' + str(i) for i in range(8)]
            pca_indicator_df = pd.DataFrame(data=principalComponents, columns=columns)

            # Normalize the PCA-transformed data
            scaled_pca_data = scaler.fit_transform(pca_indicator_df)
            scaled_pca_df = pd.DataFrame(scaled_pca_data, columns=columns)

            #################################################################################################################################
            # 
            # cdl Patterns Integration
            # 
            #################################################################################################################################

            # Lists of candlestick patterns
            bull_cdl_list = ['cdl3whitesoldiers', 'cdlmarubozu', 'cdltakuri', 'cdlmorningstar', 'cdlladderbottom',
                            'cdl3starsinsouth', 'cdl3inside', 'cdlconcealbabyswall', 'cdl3outside',
                            'cdlseparatinglines', 'cdltasukigap', 'cdlunique3river', 'cdlkicking', 
                            'cdlkickingbylength', 'cdlengulfing', 'cdlpiercing']
            condition_bull_cdl_list = ['cdlabandonedbaby', 'cdldoji', 'cdldragonflydoji', 'cdlhammer', 
                                    'cdlinvertedhammer', 'cdllongleggeddoji', 'cdlmorningdojistar', 
                                    'cdlhomingpigeon', 'cdlharamicross', 'cdlmathold']
            bear_cdl_list = ['cdl3blackcrows', 'cdldarkcloudcover', 'cdlgravestonedoji', 'cdlshootingstar',
                            'cdlclosingmarubozu', 'cdleveningstar', 'cdl2crows', 'cdlidentical3crows', 
                            'cdladvanceblock']
            condition_bear_cdl_list = ['cdlbelthold', 'cdlbreakaway', 'cdleveningdojistar', 'cdlhangingman',
                                    'cdlcounterattack', 'cdlonneck', 'cdlstalledpattern', 'cdlupsidegap2crows']
            reverse_cdl = ['cdldojistar', 'cdlharami', 'cdlhighwave', 'cdlhikkake', 'cdlhikkakemod', 'cdlinneck',
                        'cdlrickshawman', 'cdlrisefall3methods', 'cdlshortline', 'cdlspinningtop', 'cdltristar', 
                        'cdlthrusting', 'cdl3linestrike', 'cdlgapsidesidewhite']

            # Initialize a DataFrame to aggregate all patterns
            cdl_whole_df = pd.DataFrame()

            # Process all candlestick pattern lists
            all_cdl_lists = [bull_cdl_list, condition_bull_cdl_list, bear_cdl_list, condition_bear_cdl_list, reverse_cdl]
            list_names = ["bull", "condition_bull", "bear", "condition_bear", "reverse"]  # Names for each pattern list

            for idx, cdl_list in enumerate(all_cdl_lists):
                # Create a new DataFrame for the current list's data
                pattern_df = df[cdl_list].copy()

                # Sum all patterns in the list
                tmp_cdl_series = pattern_df.sum(axis=1)

                # Normalize the summed patterns
                scaler = MinMaxScaler()
                normalized_tmp_cdl_series = pd.Series(
                    scaler.fit_transform(tmp_cdl_series.values.reshape(-1, 1)).ravel(), name=list_names[idx])

                # Append the normalized data to the aggregate DataFrame
                cdl_whole_df = pd.concat([cdl_whole_df, normalized_tmp_cdl_series], axis=1)

            #################################################################################################################################
            # 
            # Integrate OHCL, date, indicators, cdl patterns
            # 
            #################################################################################################################################

            # Creating the final DataFrame by selecting specific columns to form a new DataFrame
            final_df = df[['date', 'open', 'low', 'high', 'close', 'volume']].copy()

            # Use the concat function to merge these three DataFrames along columns (axis=1)
            final_df = pd.concat([final_df, scaled_pca_df, cdl_whole_df], axis=1)

            #################################################################################################################################
            # 
            # Saving
            # 
            #################################################################################################################################

            # Save the DataFrame with indicators
            single_stock_with_indicator = stocks_indicator_path + stock_name
            final_df.to_csv(single_stock_with_indicator, index=False)
            
# Creating a thread pool and setting the maximum number of threads
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    # Get all .csv files
    csv_files = [filename for filename in os.listdir(stocks_path) if filename.endswith('.csv')]

    # Submit each .csv file for processing as a separate task to the pool
    futures = [executor.submit(process_single_stock, filename) for filename in csv_files]

    # Wait for all tasks to complete
    for future in concurrent.futures.as_completed(futures):
        try:
            # Capture the result of each task, handle exceptions if any
            result = future.result()
        except Exception as exc:
            print(f'Task generated an exception: {exc}')