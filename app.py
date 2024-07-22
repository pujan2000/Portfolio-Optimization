from flask import Flask, render_template, request, redirect, url_for
import logging
from logging.handlers import RotatingFileHandler
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize


app = Flask(__name__, static_url_path='/static')

if not app.debug:
    handler = RotatingFileHandler('error.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.ERROR)
    app.logger.addHandler(handler)

def fetch_data(tickers,years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days = years*365)
    prices = pd.DataFrame()
    for ticker in tickers:
        prices[ticker] = yf.download(ticker, start = start_date, end = end_date)["Adj Close"]
    return prices

def get_logreturns(prices):
    log_returns = pd.DataFrame()
    for i in prices:
        log_returns[i] = np.log(prices[i] / prices[i].shift())
    return log_returns

def get_annual_measures(log_returns):
    annual_returns = []
    for i in log_returns:
          annual_returns.append(np.exp(np.mean(log_returns[i]) * 252) - 1)
    annual_sd = []
    for i in log_returns:
        annual_sd.append((np.std(log_returns[i])) * np.sqrt(252))
    return annual_returns, annual_sd

def get_portfolio_measures(weights,cov_matrix,annual_returns,annual_sd):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) #Derived from summation i to n(summation j to n (wi*wj*sdi*sdj*corij))
    annual_portfolio_sd = np.sqrt(portfolio_variance) * np.sqrt(252)
    annual_portfolio_mean = np.dot(weights,annual_returns)
    return annual_portfolio_mean, annual_portfolio_sd

def Sharpe_ratio_calc(weights):
    annual_portfolio_mean, annual_portfolio_sd = get_portfolio_measures(weights,cov_matrix,annual_returns,annual_sd)
    sharpe_ratio = (annual_portfolio_mean - rfr)/annual_portfolio_sd
    return -sharpe_ratio

def maximize_Sharpe(min_weight, max_weight, weights):
    bounds = tuple((min_weight, max_weight) for asset in range(prices.shape[1]))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(Sharpe_ratio_calc, weights, method = 'SLSQP', bounds = bounds, constraints = constraints)
    optimal_weights = result.x
    return optimal_weights

def portfolio_sd(weights):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) #Derived from summation i to n(summation j to n (wi*wj*sdi*sdj*corij))
    annual_portfolio_sd = np.sqrt(portfolio_variance) * np.sqrt(252)
    return annual_portfolio_sd

def minimize_risk(min_weight,max_weight, weights):
    bounds = tuple((min_weight, max_weight) for asset in range(prices.shape[1]))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(portfolio_sd, weights, method = 'SLSQP', bounds = bounds, constraints = constraints)
    optimal_weights = result.x
    return optimal_weights

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        min_weight = float(request.form['min_weight'])
        max_weight = float(request.form['max_weight'])
        years = int(request.form['years'])
        # tickers = request.form['tickers'].split()
        global prices, log_returns, annual_returns, annual_sd, cov_matrix, weights, rfr

        if request.form['input_type'] == 'manual':
            tickers = request.form['tickers'].split()
        elif request.form['input_type'] == 'file':
            file = request.files['ticker_file']
            if file and file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
                tickers = df['Tickers'].tolist()
            else:
                return "Invalid file format. Please upload an Excel file.", 400
        
        rfr = float(request.form['rfr']) / 100

        prices = fetch_data(tickers, years)
        log_returns = get_logreturns(prices)
        annual_returns, annual_sd = get_annual_measures(log_returns)
        cov_matrix = log_returns.cov()

        weights = np.ones(len(tickers)) / len(tickers)
        optimal_weights_1 = maximize_Sharpe(min_weight, max_weight, weights)
        optimal_weights_2 = minimize_risk(min_weight, max_weight, weights)

        equal_weights_return, equal_weights_sd = get_portfolio_measures(weights,cov_matrix,annual_returns,annual_sd)
        max_sharpe_return, max_sharpe_sd = get_portfolio_measures(optimal_weights_1,cov_matrix,annual_returns,annual_sd)
        min_risk_return, min_risk_sd = get_portfolio_measures(optimal_weights_2,cov_matrix,annual_returns,annual_sd)

        equal_weights_ratio = -Sharpe_ratio_calc(weights)
        max_sharpe_ratio = -Sharpe_ratio_calc(optimal_weights_1)
        min_risk_ratio = -Sharpe_ratio_calc(optimal_weights_2)

        # plot_url = visualize(optimal_weights_1, optimal_weights_2)
        return render_template('results.html', years = years, 
                               tickers = tickers,
                               annual_returns = annual_returns,
                               annual_sd = annual_sd,

                               weights = weights,
                               optimal_weights_1 = optimal_weights_1,
                               optimal_weights_2 = optimal_weights_2, 

                               equal_weights_return = equal_weights_return,
                               equal_weights_sd = equal_weights_sd,
                               max_sharpe_return = max_sharpe_return,
                               max_sharpe_sd = max_sharpe_sd,
                               min_risk_return = min_risk_return,
                               min_risk_sd = min_risk_sd,
                               
                               equal_weights_ratio = equal_weights_ratio,
                               max_sharpe_ratio = max_sharpe_ratio,
                               min_risk_ratio = min_risk_ratio)
    except Exception as e:
        app.logger.error(f"Error during optimization: {e}")
        return "An error occurred during optimization", 500

if __name__ == '__main__':
    app.run(debug=True)

