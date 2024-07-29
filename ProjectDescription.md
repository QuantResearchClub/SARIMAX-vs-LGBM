# Project:
每个小组将创建一个由至多10只股票组成的投资组合，在纳斯达克、富时和新交所三个市场中的任何一个公开交易。所有投资组合股票必须在2024年8月1日之前交易至少一年。每个小组都需要研究能够自动更新10只股票组合交易策略的模型和方法。在2024M1 ~ 2024M6(1到6月, 作为验证期，允许策略更新)和 2022M7(测试期，固定策略，不更新) 进行性能评估。投资组合中的股票列表及其从2024年1月1日至2024年7月31日的每日收盘价必须以CSV文件的形式与项目演示文稿一起提交。
###要求:
#### 文件格式:
1. 所有上传文件为PDF格式，A4格式，纵向，四周距25mm。页边空白处不应出现任何内容;甚至不是图表的一部分。
2. Arial字体，字号12，行距1.5行(即行距介于单行和双行之间)。计算机输出，图形，表格等，可以在单一的间距，可以使用不同的字体大小，如果需要，以适应页面宽度。
3. 仅上传至CANVAS。不要把文件用电子邮件发给你的老师/导师。
4. 每组只上传1个zip文件。任何额外上传的文件将不会被读取，并将导致最高可能分数的5%的罚款。
5. 页数限制：项目报告的最大页数为6页。
### 内容：
- 第一页为执行摘要。执行摘要应包括
  1. 研究目标:
  你想做什么?清楚地说明要解决的问题，并解释为什么它对经济和/或社会很重要。
  2. 建议的方法:
  你的方法是什么?
  从你的方法中产生的科学突破或颠覆性创新的潜力是什么?
  描述以前和正在进行的工作，以及任何初步结果，提供有助于支持该提案的必要细节。
  为什么你认为你的建议能成功地解决这个问题?
  3. 团队成员的角色:
  团队成员的角色和贡献是什么?
  简要描述为实现研究目标的成员之间的互动计划。
  4. 结果:结果是什么?
    剩下的5页包括任何标题页，表格和图表(在第一页的顶部包括几行任何标题，姓名等)。
    所有的页码都要编号。任何超出限制的页面将不会获得任何分数(并且可能无法阅读);
    相反，每多出一页就要扣掉最高分数的5% !
    参考文献可以在附录中，不计算在页数限制内。
### 想法: 
- 研究一下如何拿Akshare搭建一个去拿股票数据的接口, 先拿到并获取一些交易数据。
  - Training Set:   2023M1 - 2023M12
  - Validation Set: 2024M1 - 2024M6
  - Duration: A Portfolio (10 Stocks) in price within 1 month.
  - 初步想法: 从图形上确定第二天价格是更贵还是更低, 如果第二天的价格是更改的, 就采取买入, 然后在第二天卖出。
- 然后找一篇关于Portfolio优化的论文, 按照这个论文去干, 我觉得可以试一下拿FinAgent实现, 或者再找找别的工具 (July 07)，

# 计算参数:
- `sd_m` standard deviation by month
- `sd_w` standard deviation by week.
- `bollinger_strat` 布林格参数, 用于估计股票的波动范围
- `heikin_ashi` HA, 用于估计受市场噪音的影响
- `logret`
- `stochastic_oscillator`
- `bears_bulls_power`
- `rsi`
- `logret`
- `mom`
- `obv`
- `accum_distribute`
- `aroon`
- `atr`

### 1. standard deviation
We calculate the standard deviation by months and by weeks for further calculations.

### 2. bollinger_strat
By using Bollinger_strat, we could get a rough **range of the fluctuation of the share price**. Bollinger_strat is consist of by the upper, middle and the lower lines, while these lines could be considered as the pressure line, average line and the supporting line of the share price. The share price is fluctuating between the upper line and lower line.
```math
BOLU = MA(TP, n) + m \times \sigma[TP, n]
```
```math
BOLL = MA(TP, n) - m \times \sigma[TP, n]
```
where,
- MA represents the moving average calculated by the mean value of share price during such period of time,
- $\sigma$ represents the standard deviation of such period of time,
- m = 2, parameter of the standard deviation.
- $TP = (High + Close + Open) / 3$ represents the typical price.

Reference:
- https://skilling.com/row/cn/blog/trading-articles/what-are-bollinger-bands/
- https://www.investopedia.com/trading/using-bollinger-bands-to-gauge-trends/


### 3. heikin_ashi 日本蜡烛图
The Heikin-Ashi technique averages price data to create a Japanese candlestick chart that filters out market noise. 
```
Market Noise
Noise refers to information or activity that confuses or misrepresents genuine underlying trends.
```
Heikin-Ashi provides a more smoother appearance for a easier spot of trends and reversals in contrast of directly using OPEN, HIGH LOW and CLOSE, but this also obscures gaps and some price data.
```math
HA_open = (HA_open_{-1} + HA_close_{-1}) / 2
```
```math
HA_close = (HA_open_{0} + HA_close_{0} + HA_high_{0} + HA_low_{0}) / 4
```
where,
- 0 represents the current state, -1 represents the previous state.

Reference:
- https://www.investopedia.com/terms/h/heikinashi.asp

### 4 logret (The Log returns of Time series model)
Log returns are a common way to measure the performance of an investment because they are additive over time then if we have a series of log returns for a period time, we can add them up to get the total log return for that period.
```math
log_return_t=log(price_t/price_{t-1})
```
where,
- price_t is the price of the asset at time t and price_{t-1}is the asset at time t-1

Normality
A common argument for log returns is that they are normally distributed if prices are log normally distributed. 
```math
z~N(0,1)
x=exp(μ+σz)
log(x)~N(μ,σ^2)
z_t=log(p_t)-log(p_t-1)
```
The advantages of log returns are:
- 1.Additive
- 2.Log returns are symmetric around zero
- 3.Log returns are normally distributed
- 4.Log returns simplify complex calculations

Compare to return
Returns are lower-bounded by -1. One cannot lose more than all of one’s money. However, log returns have an infinite support. And since the log function suppresses big positive values while emphasizing small negative values, log returns are more symmetric than returns. 
Using simple returns for time series analysis can lead to misleading insights, especially when dealing with long-term data and significant price fluctuations.

Simple returns:
1. Analyzing short-term investment performance
2. Dealing with small changes in asset prices
3. Communicating financial data to a general audience

Log returns:
1. Analyzing long-term investment performance
2. Working with continuous time series data and conducting mathematical modeling
3. Seeking accuracy in compounding effects over time

In time series analysis logrithms is often considered to stabilize the variance of a series. For a range of economic variables substantial forecasting improvements from taking logs are found if the log transformation actually stabilizes the variance of the underlying series.

Strategy:
1. Mean Reversion Strategy
2. Momentum Trading Strategy
3. Trend Following Strategy
4. Volatility Breakout Strategy
5. Risk Parity Strategy

Reference:
- https://medium.com/quant-factory/why-we-use-log-return-in-finance-349f8a2dc695
- LUETKEPOHL, Helmut, XU, Fang, The Role of log Transformation in Forecasting Economic Variables, EUI MWP, 2009/06 - https://hdl.handle.net/1814/11150
- https://gregorygundersen.com/blog/2022/02/06/log-returns/
- https://medium.com/@manojkotary/simple-returns-vs-log-returns-a-comprehensive-comparative-analysis-for-financial-analysis-702403693bad


### 3 stochastic_oscillator
A stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result. 
- A technical indicator for generating overbought and oversold signals
- Tend to vary around some mean price level since they rely on an asset's price history
- Measure the momentum of an asset's price to determine trends and predict reversals
- Measure recent prices on a scale of 0 to 100, with measurements above 80 indicating that an asset is overbought and measurements below 20 indicating that it is oversold.

Overbought (超买) refers to a security that trades at a price level above its true (or intrinsic) value. Traders may expect a price correction or trend reversal, so they may sell the security

Oversold (超卖) refers to a security that trades at a price level below its true value (lower than it should be). Traders may expect a price correct or trend reversal, so they may buy the security

Stochastic oscillator charting generally consists of two lines: one reflecting the actual value of the oscillator for each session (%K) and one reflecting its three-day simple moving average (%D). These two lines are used to show the relationship between current and past prices. Because price is thought to follow momentum, the intersection of these two lines is considered to be a signal that a reversal may be in the works, as it indicates a large shift in momentum from day to day.

Stochastics oscillator and the Relative Strength Index (RSI—another popular technical indicator) “capture” these wave-like price swings within certain limits, allowing you to measure the strength or “momentum” of these fluctuation

RSI: 
The relative strength index (RSI) is a momentum indicator used in technical analysis. 

Difference between RSI and stochastic oscillator:

The stochastic oscillator is predicated on the assumption that closing prices should close in the same general direction as the current trend. RSI tracks overbought and oversold levels by measuring the velocity of price movements. But both of them are used as overbought/oversold indicators.

```math
%K = (Last closing price - lowest price)/(Highest price - lowest price) * 100
%D = 3 day SMA of %K
```
where,
- C is the last closing price
- Lowest price for the time period
- Highest prie for the time period
- SMA is the simple moving average (the average price over the specified period)

Use of the Stochastic Oscillator:
1. Identify overbought and oversold levels

   - Stochastic reading > 80 then overbought; Stochastic reading < 20 then oversold
   - A sell signal is generated when the oscillator reading > 80 and then returns to < 80. A buy signal is indicated when the oscillator < 20 and then > 20. 
   - Overbought and oversold levels mean that the security’s price is near the top or bottom, respectively, of its trading range for the specified time period.

2. Divergence
   - Divergence occurs when the security price is making a new high or low that is not reflected on the Stochastic Oscillator. Please note that the Stochastic Oscillator may give a divergence signal some time before price action changes direction.

   eg. price moves to a new high but the oscillator does not move to the new high reading correspondingly, which may signal an impending market reversal from an uptrend to a downtrend. Same with the new low case

3. Crossovers
   - Crossovers refer to the point at which the fast stochastic line and the slow stochastic line intersect.
   - Fast stochastic line: %K ; Slow stochastic line: %D 

   When the %K line intersects the %D line and goes above it, this is a bullish scenario. Conversely, the %K line crossing from above to below the %D stochastic line gives a bearish sell signal.

Limitation:

The main shortcoming of the oscillator is its tendency to generate false signals. Especially during turbulent, highly volatile trading conditions. 

Key idea: 

Oscillator is primarily designed to measure the strength or weakness – not the trend or direction – of price action movement in a market.

Reference:
- https://www.investopedia.com/terms/s/stochasticoscillator.asp
- https://www.britannica.com/money/stochastic-oscillator-technical-indicator
- https://www.investopedia.com/terms/r/rsi.asp
- https://www.investopedia.com/ask/answers/012015/what-are-differences-between-relative-strength-index-rsi-stochastic-oscillator.asp
- https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/stochastic-oscillator/

### 4 bears_bulls_power
The Bulls Power indicator is telling you how strong the bulls are (the buyers). The Bears Power indicator is telling you how strong the bears are (the sellers).
```math
Bulls Power =  High - EMA Bears Power = Low - EMA
```
where
- EMA is the exponential moving average, use the 13-day EMA commonly

Bulls Power and Bears Power indicators is the distance between the EMA and the high/low. This corresponds with the influence of the bulls/bears to continue to push price past the EMA. Note that positive values in the Bulls Power indicator indicate bullish strength, while negative values in the Bears Power indicator indicate bearish strength.

Strategy:
- Buy:
   - The EMA is rising
   - Bears Power is negative but moving closer to 0 
- Sell: 
   - The EMA is falling
   - Bulls Power is positive but moving toward 0 

EMA: 

Exponential Moving Average (EMA) is similar to Simple Moving Average (SMA), measuring trend direction over a period of time. However, whereas SMA simply calculates an average of price data, EMA applies more weight to data that is more current.

- Calculation
   1. Calculate simple moving average (SMA)
```math
(Sum of closing prices of the specified time period)/(Number of observations)
```
   2. Calculate Multiplier
```math
Multiplier = {2/(Total number observations + 1)}
```
   3. Calculate exponential moving average (EMA)
```math
EMA = {Closing price of the stock * Multiplier} + {previous day's EMA * (1-Multiplier)}
```
- Advantages
   1. Identifies and confirms market trends
   2. Determine support and resistance levels
   3. More sensitive

Reference:
- https://www.earnforex.com/guides/beginners-guide-to-the-bulls-power-and-bear-power-indicators-in-forex/
- https://www.wallstreetoasis.com/resources/skills/trading-investing/exponential-moving-average-ema?gad_source=1&gclid=CjwKCAjw2Je1BhAgEiwAp3KY79ju6KkQcfLdgPu15O9FdI3_hygk4jLNwRRio2BvMarE9wi9YwyDlBoCCY8QAvD_BwE

### 5 rsi (Relative Strength Index)
The relative strength index (RSI) is a momentum indicator used in technical analysis. RSI measures the speed and magnitude of a security's recent price changes to evaluate overvalued or undervalued conditions in the price of that security.

The RSI can  point to overbought and oversold securities and indicate securities that may be primed for a trend reversal or corrective pullback in price. It can signal when to buy and sell. Traditionally, an RSI reading of 70 or above indicates an overbought situation. A reading of 30 or below indicates an oversold condition.
```math
RSI_{step1} = 100-[100/(1+(Average gain/Average loss))]
```
where,
- The average gain or loss are the average percentage gain or loss during the period and both of them take positive value
- Price decreasing period counted as zero in average gain; Price increasing period counted as zero in average loss
- The standard number of periods used to calculate the initial RSI value is 14

Once 14 periods of data available then move to the second step of calculation
```math
RSI_{step2} = 100-[100/[1+(((Previous Average Gain * 13)+Current Gain)/((Previous Average Loss * 13)+Current Loss))]]
```

Plotting RSI
The RSI will rise as the number and size of up days increase. It will fall as the number and size of down days increase.

Why important:

1. predict the price behavior of a security
2. validate trends and trend reversals
3. point to overbought and oversold securities
4. provide short-term traders with buy and sell signals

Investors create a horizontal trendline between the levels of 30 and 70 when a strong trend is in place to better identify the overall trend and extremes. Since the RSI is not as reliable in trending markets as it is in trading ranges. In addition, the singals given by the RSI is strong upward or downward trends often can be false

Strategy:
1. Overbought and Oversold
2. Divergence : An RSI divergence occurs when price moves in the opposite direction of the RSI
   - A bullish divergence occurs when the RSI displays an oversold reading (< 30) followed by a higher low that appears with lower lows in the price. 
   - A bearish divergence occurs when the RSI creates an overbought reading (> 70) followed by a lower high that appears with higher highs on the price.
3. RSI Midline Strategy
   - Sell: When RSI has stayed above the 50 for awhile, go short when RSI closes below the 50 level
   - Buy: When RSI has stayed below the 50 for awhile, go long when RSI closes above the 50 level
4. RSI Exit Trading Strategy
   - Sell: When the RSI gets above 70, sell after RSI closes below the 70 level
   - Buy: When RSI gets below 30, buy after RSI closes above the 30 level
5. Larry Connors RSI 2 Trading Strategy
   - Sell: When RSI is overbought and price closes below the 200 SMA
   - Buy: When RSI is oversold and price closes above the 200 SMA
   
Reference:
- https://www.investopedia.com/terms/r/rsi.asp
- https://www.tradingheroes.com/rsi-trading-strategies/

### 6 mom (momentum)

Momentum is the speed or velocity of price changes in a stock, security, or tradable instrument. Momentum shows the rate of change in price movement over a period of time to help investors determine the strength of a trend. 

Momentum indicator and add it to the given dataset. Specifically, it calculates the sum of the closing prices over a specified window period. The momentum compares the current closing price with the previous closing price for n periods ago and gives a digital conclusion, according to which a momentum trader determines whether it is profitable to buy or sell, what is a potential profit for a trade, and whether a trend should reverse soon.

Idea: A stock can be exhibit bullish momentum, meaning the price is rising, or bearish momentum where the price is steadily falling
```math
Momentum = V-V_x
```
where,
- V = Lastest price
- V_x = Closing price
- x = Number of days ago

Mesuring Momentum:
1. typically use a 10-day time frame when measuring
2. If the most recent closing price of the index is more than the closing price 10 trading days ago then the positive number will be plotted above the zero line
3. If the lastest closing price is lower than the closing price 10 days ago, then the negative number will be plotted below the zero line
4. The zero line is essentially an area where the index or stock is likely trading sideways or has no trend
5. Once the stock's monmentum has increase (no matter bullish or bearish), the momentum line moves farther away from the zero line

Special cases:
1. When the momentum indicator slides below the zero line and then reverses ina n upward direction, it means downtrend is slowing down but not dowtrend is over.

strategy:
1. When the momentum indicators’ line breaks out the zero level upside, there is a potential buy signal and When the momentum indicators’ lines break out the zero line downside, there is a potential sell signal

Reference:
- https://www.litefinance.org/blog/for-beginners/best-technical-indicators/momentum-indicator/
- https://www.investopedia.com/articles/technical/081501.asp


