import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

israel_attack_iran = datetime(2025, 6, 13)
us_attack_iran = datetime(2025, 6, 22)
ceasefire = datetime(2025, 6, 24)
end = datetime(2025, 7, 1)

def analyze_and_plot(ticker, period):

    commodity_df = yf.download(ticker, start=israel_attack_iran - timedelta(days=period),end=end)

    events = [("Israel_attack_iran", israel_attack_iran), ("US_attack_iran",us_attack_iran),("Ceasefire",ceasefire)]

    fig, axs = plt.subplots(1,3, figsize=(15,6))
    fig.suptitle(f"{ticker} Prices and Prediction Before & After Conflict", fontsize=16, fontweight='semibold')
    for i,(event_name, event_date) in enumerate(events):

        trainingset_period1 = commodity_df[commodity_df.index < event_date]
        testingset_period1 = commodity_df.drop(trainingset_period1.index)
        X_train = np.arange(len(trainingset_period1)).reshape(-1, 1)
        Y_train =trainingset_period1["Close"].values

        model = LinearRegression()
        model.fit(X_train, Y_train)
        X_test = np.arange(len(trainingset_period1), len(trainingset_period1) + len(testingset_period1)).reshape(-1,1)
        Y_test = model.predict(X_test)

        #Model Statistics
        r2 = model.score(X_train,Y_train)
        slope = model.coef_[0][0]
        Y_Pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(Y_train,Y_Pred))
        
        #Plotting
        axs[i].plot(commodity_df.index, commodity_df["Close"], label="Actual Close Price")
        axs[i].plot(testingset_period1.index, Y_test, label="Predicted Close Price", linestyle='--')
        axs[i].plot(trainingset_period1.index, Y_Pred, linestyle='--', color="black")
        axs[i].axvline(x=event_date, color='red', linestyle=':', label=event_name)
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel("Price (USD)")
        axs[i].legend()
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        axs[i].grid(True)
        axs[i].set_title(f"{event_name}\nR² = {r2:.3f}  |  Slope = {slope:.2f}  |  RMSE = {rmse:.2f}", fontsize=10)
        
    return commodity_df,fig