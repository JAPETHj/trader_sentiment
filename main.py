import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data(sentiment_file, trade_file):

    sentiment_df = pd.read_csv(sentiment_file)
    trades_df = pd.read_csv(trade_file)

    print(f"Sentiment Dataset Shape: {sentiment_df.shape}")
    print(f"Trades Dataset Shape: {trades_df.shape}")

    return sentiment_df, trades_df


def preprocess_data(sentiment_df, trades_df):

    sentiment_df = sentiment_df.rename(
        columns={
            'date': 'Date',
            'classification': 'Classification'
        }
    )

    sentiment_df['Date'] = pd.to_datetime(
        sentiment_df['Date']
    ).dt.normalize()

    trades_df = trades_df.rename(
        columns={
            'Account': 'account',
            'Coin': 'symbol',
            'Execution Price': 'execution_price',
            'Size Tokens': 'size_tokens',
            'Size USD': 'size',
            'Side': 'side',
            'Start Position': 'start_position',
            'Direction': 'direction',
            'Closed PnL': 'closedPnL',
            'Fee': 'fee'
        }
    )

    trades_df['time'] = pd.to_datetime(
        trades_df['Timestamp'],
        unit='ms'
    )

    trades_df['Date'] = (
        trades_df['time']
        .dt.normalize()
    )

    trades_df['closedPnL'] = pd.to_numeric(
        trades_df['closedPnL'],
        errors='coerce'
    )

    print(
        "\nSentiment Range:",
        sentiment_df['Date'].min(),
        "->",
        sentiment_df['Date'].max()
    )

    print(
        "Trade Range:",
        trades_df['Date'].min(),
        "->",
        trades_df['Date'].max()
    )

    common_dates = (
        set(sentiment_df['Date'])
        &
        set(trades_df['Date'])
    )

    print(
        "Common Dates:",
        len(common_dates)
    )

    return sentiment_df, trades_df

def merge_data(sentiment_df, trades_df):

    merged_df = pd.merge(
        trades_df,
        sentiment_df[['Date', 'Classification']],
        on='Date',
        how='left'
    )

    merged_df = merged_df.dropna(
        subset=['Classification']
    )

    merged_df = merged_df.dropna(
        subset=['closedPnL']
    )

    merged_df['Win'] = (
        merged_df['closedPnL'] > 0
    ).astype(int)

    sentiment_map = {
        'Extreme Fear': 0,
        'Fear': 25,
        'Neutral': 50,
        'Greed': 75,
        'Extreme Greed': 100
    }

    merged_df['SentimentScore'] = (
        merged_df['Classification']
        .map(sentiment_map)
    )

    print(
        f"\nMatched Trades: {len(merged_df):,}"
    )

    return merged_df

def sentiment_analysis(merged_df):

    summary = merged_df.groupby(
        'Classification'
    ).agg(
        Trade_Count=('closedPnL', 'count'),
        Total_PnL=('closedPnL', 'sum'),
        Avg_PnL=('closedPnL', 'mean'),
        Median_PnL=('closedPnL', 'median'),
        Win_Rate=('Win', 'mean')
    ).reset_index()

    summary['Win_Rate'] *= 100

    print("\n===== SENTIMENT ANALYSIS =====")
    print(summary)

    summary.to_csv(
        "sentiment_analysis.csv",
        index=False
    )

    return summary

def plot_sentiment_analysis(summary):

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary,
        x='Classification',
        y='Avg_PnL'
    )
    plt.title(
        'Average PnL by Sentiment'
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary,
        x='Classification',
        y='Win_Rate'
    )
    plt.title(
        'Win Rate by Sentiment'
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary,
        x='Classification',
        y='Total_PnL'
    )
    plt.title(
        'Total PnL by Sentiment'
    )
    plt.tight_layout()
    plt.show()

def long_short_analysis(merged_df):

    if 'side' not in merged_df.columns:
        return

    side_analysis = merged_df.groupby(
        ['Classification', 'side']
    ).agg(
        Trades=('closedPnL', 'count'),
        Avg_PnL=('closedPnL', 'mean'),
        Total_PnL=('closedPnL', 'sum')
    ).reset_index()

    print("\n===== LONG VS SHORT =====")
    print(side_analysis)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=side_analysis,
        x='Classification',
        y='Avg_PnL',
        hue='side'
    )
    plt.title(
        'Long vs Short Performance'
    )
    plt.show()

    return side_analysis

def symbol_analysis(merged_df):

    if 'symbol' not in merged_df.columns:
        return

    symbol_stats = merged_df.groupby(
        ['symbol', 'Classification']
    ).agg(
        AvgPnL=('closedPnL', 'mean'),
        TotalPnL=('closedPnL', 'sum'),
        Trades=('closedPnL', 'count')
    ).reset_index()

    print("\n===== TOP SYMBOLS =====")

    top_symbols = (
        symbol_stats.groupby('symbol')
        ['TotalPnL']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print(top_symbols)

    pivot = symbol_stats.pivot(
        index='symbol',
        columns='Classification',
        values='AvgPnL'
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        cmap='RdYlGn'
    )

    plt.title(
        'Symbol vs Sentiment'
    )

    plt.show()

    return symbol_stats


def consistency_analysis(merged_df):

    consistency = merged_df.groupby(
        'account'
    ).agg(
        MeanPnL=('closedPnL', 'mean'),
        StdPnL=('closedPnL', 'std')
    )

    consistency = consistency.dropna()

    consistency['ConsistencyScore'] = (
        consistency['MeanPnL']
        /
        consistency['StdPnL']
    )

    consistency = consistency.sort_values(
        by='ConsistencyScore',
        ascending=False
    )

    print(
        "\n===== MOST CONSISTENT TRADERS ====="
    )

    print(consistency.head(20))

    return consistency



def monthly_analysis(merged_df):

    temp = merged_df.copy()

    temp['Month'] = (
        temp['Date']
        .dt.to_period('M')
    )

    monthly = temp.groupby(
        ['Month', 'Classification']
    ).agg(
        TotalPnL=('closedPnL', 'sum')
    ).reset_index()

    monthly['Month'] = (
        monthly['Month']
        .astype(str)
    )

    plt.figure(figsize=(14, 6))

    sns.lineplot(
        data=monthly,
        x='Month',
        y='TotalPnL',
        hue='Classification'
    )

    plt.xticks(rotation=45)

    plt.title(
        'Monthly Profitability by Sentiment'
    )

    plt.show()

    return monthly


def main():

    sentiment_file = "fear_greed_index.csv"
    trade_file = "historical_data.csv"
    
    sentiment_df, trades_df = load_data(
        sentiment_file,
        trade_file
    )

    sentiment_df, trades_df = preprocess_data(
        sentiment_df,
        trades_df
    )

    merged_df = merge_data(
        sentiment_df,
        trades_df
    )

    summary = sentiment_analysis(merged_df)

    plot_sentiment_analysis(summary)

    long_short_analysis(merged_df)

    symbol_analysis(merged_df)

    consistency_analysis(merged_df)

    monthly_analysis(merged_df)

    merged_df.to_csv(
        "merged_trader_sentiment_data.csv",
        index=False
    )

    print("\nAnalysis Completed Successfully!")


if __name__ == "__main__":
    main()