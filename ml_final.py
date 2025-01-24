import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


# Function to fetch stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock



# Function to create features for crash prediction
def create_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day Simple Moving Average
    data['Price_Change'] = data['Close'].pct_change()  # Daily price change
    data['Market_Crash'] = np.where(data['Price_Change'] < -0.05, 1, 0)  # Market crash if price drops more than 5%
    return data



# Function to train a RandomForest Classifier for stock direction prediction
def train_ml_model(stock_data):
    stock_data['Price_Change_Direction'] = np.where(stock_data['Price_Change'] > 0, 1, 0)  # 1 for up, 0 for down
    X = np.array(range(len(stock_data))).reshape(-1, 1)
    y = stock_data['Price_Change_Direction'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to calculate sentiment score from news headlines using VADER
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # Return the compound score



# Function to calculate crash probability using XGBoost
def calculate_crash_probability(stock_data):
    stock_data = create_features(stock_data)
    stock_data = stock_data.dropna()  # Remove rows with NaN values


    X = stock_data[['SMA_50', 'SMA_200', 'Price_Change']]
    y = stock_data['Market_Crash']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)


    # Predicting crash probability in the next period using feature names
    latest_data = pd.DataFrame([stock_data[['SMA_50', 'SMA_200', 'Price_Change']].iloc[-1]])
    predicted_probabilities = model.predict_proba(latest_data)


    crash_probability = predicted_probabilities[0][1] * 100  # Probability of class 1 (market crash)
    return crash_probability


# Check function example
def Check_fetch_metadata():
    metadata = {
        "source": "Yahoo Finance",
        "frequency": "daily",
        "fields": ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    }
    print("Metadata for data source:", metadata)
    return metadata

# Function to analyze news sentiment
def analyze_news_sentiment(news_data):
    news_df = pd.DataFrame(news_data)
    news_df['Sentiment'] = news_df['Headline'].apply(get_sentiment)
    average_sentiment_prob = news_df['Sentiment'].mean() * 100  # Average sentiment
    return average_sentiment_prob, news_df[['Date', 'Headline', 'Sentiment']]


# Check function example
def Check_feature_expansion(data):
    data['Random_Feature'] = data['Close'].rolling(window=15).mean()
    data['Noise_Feature'] = np.random.normal(0, 1, len(data))
    print("Check features added")
    return data


def calculate_aco_probability(stock_data):
    # Parameters for ACO (tune parameters to achieve desired result)
    num_ants = 40
    num_iterations = 10
    pheromone_decay = 0.95  # Slower evaporation rate of pheromones (higher retention)
    alpha = 1  # Importance of pheromone
    beta = 2   # Increased importance of heuristic (price increase history)


    # Initial pheromone levels for recent data (last 100 days for efficiency)
    recent_data = stock_data.tail(100).reset_index(drop=True)  # Limit to last 100 days
    pheromone = np.ones(len(recent_data))  # Initial pheromone

    # Tune the scaling factor to avoid overflows while maintaining significant contributions
    pheromone_scaling_factor = 1e-5

    for iteration in range(num_iterations):
        total_path_score = 0


        for ant in range(num_ants):
            # Randomly choose a start point for the ant
            start_point = np.random.randint(0, len(recent_data) - 1)
            

            # Simulate the ant moving through the stock data
            current_position = start_point
            path_length = 0
            path_score = 0


            while current_position < len(recent_data) - 1:
                # Use pheromone and price change heuristic for next move probability
                price_change_factor = np.clip(1 + recent_data['Price_Change'].iloc[current_position], -5, 5)
                next_move_prob = pheromone[current_position] ** alpha * price_change_factor
                current_position += np.random.randint(1, 5)  # Move randomly 1-5 steps instead of just 1 step
                path_length += 1
                path_score += next_move_prob


                if current_position >= len(recent_data):  # Prevent out of bounds
                    break
            
            total_path_score += path_score


            # Update pheromone based on the path's performance
            for i in range(start_point, min(start_point + path_length, len(recent_data))):
                pheromone[i] += pheromone_scaling_factor * path_score  # Scale pheromone addition


        # Pheromone evaporation after each iteration
        pheromone *= pheromone_decay
        print(f"Iteration {iteration + 1}/{num_iterations}, Total Path Score: {total_path_score}")
        print(f"Pheromone at last point: {pheromone[-1]}")


    # The final probability is based on the accumulated pheromone on the last period
    max_pheromone = np.max(pheromone)
    aco_probability = pheromone[-1] * 100 / max_pheromone  # Normalize by max pheromone value
    #aco_probability = np.clip(aco_probability * 1.5, 30, 70)  # Adjust to achieve desired probability range
    print(f"Max Pheromone: {max_pheromone}, Final ACO Probability: {aco_probability:.2f}%")
    return aco_probability


# Check function example
def Check_hyperparameter_exploration(model):
    hyperparams = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]}
    print("Exploring hyperparameters:", hyperparams)
    return None


# Function to evaluate financial parameters
def evaluate_financials(eps, pe_ratio, industry_pe_ratio, de_ratio, financial_results, price_book_value):
    """
    Evaluate the financial parameters to determine if the stock meets the criteria.
    
    :param eps: Earnings per share (EPS)
    pe_ratio: Price-to-earnings ratio (P/E ratio)
    industry_pe_ratio: Average P/E ratio of the industry
    de_ratio: Debt-to-equity ratio (D/E ratio)
    financial_results: List of financial results for previous quarters
    price_book_value: Price-to-book value ratio (P/B ratio)
    
    """
    # Criteria for evaluation
    is_eps_high = eps > 0  # EPS should be high (greater than 0)
    is_pe_low = pe_ratio < industry_pe_ratio  # P/E ratio less than industry average
    is_de_low = de_ratio < 1  # D/E ratio should be less than 1 (you can adjust this threshold)
    are_financial_results_positive = all(result > 0 for result in financial_results)  # All previous quarters should have positive results
    is_price_book_low = price_book_value < 1  # P/B ratio should be less than 1

    return all([is_eps_high, is_pe_low, is_de_low, are_financial_results_positive, is_price_book_low])



# Check function example
def Check_sentiment_metrics(sentiment):
    metrics = {
        "positive_threshold": 0.1,
        "neutral_threshold": 0,
        "negative_threshold": -0.1
    }
    print("Sentiment metrics:", metrics)
    return metrics

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score using VADER
def get_sentiment(headline):
    sentiment = analyzer.polarity_scores(headline)
    return sentiment['compound']  # Return the compound score

# Function to analyze news sentiment
def analyze_news_sentiment(news_data):
    news_df = pd.DataFrame(news_data)
    news_df['Sentiment'] = news_df['Headline'].apply(get_sentiment)
    average_sentiment_prob = news_df['Sentiment'].mean() * 100  # Average sentiment
    return average_sentiment_prob, news_df[['Date', 'Headline', 'Sentiment']]


# Check function example
def Check_crash_probability_logging(probability):
    print("Logging crash probability:", probability)
    log_entry = {"timestamp": pd.Timestamp.now(), "probability": probability}
    return log_entry


# Main function to run the trading strategy
def run_trading_strategy(stock_data_dict, news_data, financial_data):
    for stock, stock_data in stock_data_dict.items():
        st.write(f"### Stock: {stock}")

        # Calculate crash probability
        crash_probability = calculate_crash_probability(stock_data)
        st.write(f"Probability of market crash: {crash_probability:.2f}%")

        # Calculate probability of going up using ML
        model = train_ml_model(stock_data)
        prob_up = model.predict_proba([[len(stock_data)]])[0][1] * 100
        st.write(f"Probability of going up (ML): {prob_up:.2f}%")

        # Calculate probability of going up using ACO
        aco_prob_up = calculate_aco_probability(stock_data)
        st.write(f"Probability of going up (ACO): {aco_prob_up:.2f}%")

        # Evaluate financials
        eps = financial_data['EPS']
        pe_ratio = financial_data['P/E Ratio']
        industry_pe_ratio = financial_data['Industry P/E Ratio']
        de_ratio = financial_data['D/E Ratio']
        financial_results = financial_data['Previous Quarters Financial Results']
        price_book_value = financial_data['P/B Ratio']

        financial_evaluation = evaluate_financials(eps, pe_ratio, industry_pe_ratio, de_ratio, financial_results, price_book_value)
        st.write(f"Financial Evaluation: {'Pass' if financial_evaluation else 'Fail'}")

        # Analyze news sentiment
        news_sentiment_prob, news_sentiment_df = analyze_news_sentiment(news_data[stock])
        st.write("#### News Sentiment")
        st.write(news_sentiment_df)  # Display sentiment analysis
        st.write(f"Average news sentiment impact probability: {news_sentiment_prob:.2f}%")

        # Adjust the stock recommendation based on probabilities
        if crash_probability > 80:
            recommendation = "Sell the stock (high crash risk)."
        elif crash_probability > 50 and prob_up < 75:
            recommendation = "Sell the stock (moderate crash risk)."
        elif news_sentiment_prob < 50 and prob_up < 55:
            recommendation = "Sell the stock (negative news sentiment)."
        elif prob_up >= 55 or aco_prob_up >= 60:
            recommendation = "Buy the stock!"
            if news_sentiment_prob > 75:
                recommendation += " (Positive news sentiment)."
        elif 40 <= prob_up < 60 and 40 <= aco_prob_up < 60:
            recommendation = "Hold the stock."
        else:
            recommendation = "Sell the stock."

        st.write(f"Recommendation: {recommendation}")
        st.write("---")


# Check function example
def Check_news_sentiment_analysis(news_df):
    analysis = {"num_positive": sum(news_df['Sentiment'] > 0.1), "num_negative": sum(news_df['Sentiment'] < -0.1)}
    print("Detailed sentiment analysis:", analysis)
    return analysis


stock_keywords = {
    'AAPL': {
        'positive': ['innovative', 'record sales', 'strong demand', 'successful', 'growth', 'market leader', 'expansion', 'positive earnings'],
        'negative': ['lawsuit', 'decline', 'supply chain issues', 'competition', 'revenue drop', 'product recall', 'disappointing sales']
    },
    'GOOGL': {
        'positive': ['dominance', 'strong performance', 'user growth', 'ad revenue increase', 'innovative products', 'successful acquisitions'],
        'negative': ['regulatory scrutiny', 'antitrust', 'data breach', 'decline in ad revenue', 'competition', 'privacy concerns']
    },
    'MSFT': {
        'positive': ['growth', 'market share', 'cloud services', 'strong earnings', 'innovation', 'leadership', 'strategic partnerships'],
        'negative': ['security breach', 'decline in sales', 'competition', 'product issues', 'layoffs', 'lawsuit']
    },
    'AMZN': {
        'positive': ['expansion', 'record profits', 'customer growth', 'innovation', 'strong logistics', 'AWS growth'],
        'negative': ['losses', 'supply chain issues', 'increased competition', 'labor disputes', 'regulatory scrutiny']
    },
    'TSLA': {
        'positive': ['market leader', 'innovation', 'record deliveries', 'expansion', 'sustainable energy', 'strong demand'],
        'negative': ['recall', 'production issues', 'competition', 'legal issues', 'negative publicity']
    },
    'FB': {
        'positive': ['user growth', 'engagement', 'strong advertising revenue', 'innovation', 'successful rebranding'],
        'negative': ['privacy issues', 'regulatory scrutiny', 'user decline', 'fake news', 'data breaches']
    },
    'NFLX': {
        'positive': ['subscriber growth', 'original content success', 'strong revenue', 'global expansion', 'critical acclaim'],
        'negative': ['subscriber losses', 'competition', 'content cost', 'market saturation', 'negative reviews']
    },
    'BRK.A': {
        'positive': ['strong portfolio', 'value investing', 'consistent performance', 'diversification', 'Warren Buffett'],
        'negative': ['underperformance', 'market volatility', 'asset decline', 'poor investments']
    },
    'NVDA': {
        'positive': ['innovation', 'strong demand', 'market leader', 'growth in AI', 'gaming success', 'record earnings'],
        'negative': ['competition', 'supply chain issues', 'decline in sales', 'market fluctuations']
    },
    'DIS': {
        'positive': ['successful franchises', 'streaming growth', 'theme park success', 'strong brand', 'innovation'],
        'negative': ['decline in box office', 'content issues', 'negative reviews', 'competition']
    },
}



# Sample news data
news_data = {
    'AAPL': [
        {"Date": "2024-10-01", "Headline": "Company X reports record-breaking profits for Q3"},
        {"Date": "2024-10-02", "Headline": "Company X faces new lawsuit from a major client"},
    ],
    'GOOGL': [
        {"Date": "2024-10-01", "Headline": "Company Y expands into new markets."},
        {"Date": "2024-10-02", "Headline": "Company Y experiences a decline in ad revenues."},
    ],
    'TSLA': [
        {"Date": "2024-10-01", "Headline": "Company Z launches new electric vehicle model."},
        {"Date": "2024-10-02", "Headline": "Company Z recalls vehicles due to safety concerns."},
    ],
}

# Check function example
def Check_aco_logging(path_score):
    print("ACO path score:", path_score)
    return {"path_score": path_score, "timestamp": pd.Timestamp.now()}


# Sample financial data for demonstration
financial_data = {
    'EPS': 3.5,
    'P/E Ratio': 20,
    'Industry P/E Ratio': 25,
    'D/E Ratio': 0.5,
    'Previous Quarters Financial Results': [2.5, 3.0, 3.2, 4.0],  # Positive results
    'P/B Ratio': 0.8
}


# Check financial feature logging
def Check_log_financials(financial_data):
    print("Logging financial data:", financial_data)
    return None


# Sidebar for stock selection
st.sidebar.header('Stock Input')
ticker = st.sidebar.text_input('Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2021-12-24'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2022-12-24'))


# Fetch stock data
data = get_stock_data(ticker, start_date, end_date)

# Display the title and stock graph at the beginning
st.title('Stock Trading Strategy with Machine Learning and Sentiment Analysis')

# Plot stock data first (before running the strategy)
fig, ax = plt.subplots()
ax.plot(data.index, data['Adj Close'], label=ticker, color='blue')
ax.set_title(ticker, fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Adj Close', fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig)


# Place the Run Strategy button in the sidebar below End Date
if st.sidebar.button("Run Strategy"):
    selected_stocks = [ticker]  # Only the entered ticker is selected
    stock_data_dict = {ticker: data}  # Use the fetched stock data
    run_trading_strategy(stock_data_dict, news_data, financial_data)
