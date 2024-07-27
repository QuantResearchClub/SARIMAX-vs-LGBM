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


