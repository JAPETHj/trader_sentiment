This project aims to answer the following questions:
Do traders perform better during Fear or Greed market conditions?
How does market sentiment impact profitability and win rates?
Are long positions more successful than short positions during bullish periods?
Which assets perform best under different sentiment regimes?
Are there traders who consistently outperform regardless of market conditions?
Can sentiment be used as an additional signal for trading decisions?

Before analysis, the following preprocessing steps were performed:
Cleaned and standardized column names
Converted timestamps into datetime format
Extracted trading dates from timestamps
Converted profit/loss and trade size columns into numeric values
Removed invalid and missing records
Merged both datasets using the trade date and sentiment date

KEY FINDINGS:
Market Sentiment Has a Significant Impact
    Trader profitability varies noticeably across different sentiment regimes.
    Market conditions play an important role in trading outcomes.
Greed Periods Tend to Be More Profitable
    Traders generally achieve higher profits during Greed and Extreme Greed periods.
    Bullish momentum appears to support stronger trading performance.
Extreme Fear Is More Challenging
    Fear-driven markets often show lower win rates.
    Increased uncertainty can make profitable trading more difficult.
Long Positions Benefit During Bullish Markets
    Long trades typically perform better during Greed and Extreme Greed periods.
    Positive market sentiment often aligns with upward price trends.
A Small Group of Traders Drives Most Profits
    A limited number of accounts contribute a large share of total profitability.
    Consistent traders are able to maintain performance across varying market conditions.

PLOTS:
Average PnL by Sentiment
Win Rate by Sentiment
Total PnL by Sentiment
Long vs Short Performance Comparison
Symbol Performance Heatmaps
Monthly Profitability Trends
