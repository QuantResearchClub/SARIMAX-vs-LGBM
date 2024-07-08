## 1. 随机变量与样本空间
$X$ 是随机变量。
$S_{X}$记作样本空间, 代表随机变量 $X$ 的取值范围。
当我们讨论随机变量的概率密度函数(pdf)的时候， 有用到 $v$ 和 $\lambda$ ，其中v 是尺度参数而 $\lambda$ 是形状参数。

![image](https://github.com/QuantResearchClub/DSA5205/assets/34060865/d186bd69-2d2c-46b2-9721-2c2e5cc06cd9)

尺度参数 Scale Parameter: 
  - 控制分布函数在幅度上的变化, 比如$\sigma$ 作为随机变量$X$的一个尺度参数有 $\sigma(aX) = |a|\sigma(X)$
  - If $\lambda$ is a precision, then $\lambda^{-1}$ is an inverse-scale parameter and quantifies precision.

形状参数 Shape Parameter:
  - 形状参数控制分布函数形状的变化，在下图中, 我们可以比较清晰地观察到两者的区别。我们有$$\lambda^{-1} f(\lambda^{-1} (y - \theta)) = f(y)$$
![image](https://github.com/QuantResearchClub/DSA5205/assets/34060865/b5e0f3a2-e7a9-48d3-8d6c-ad49abb4d701)

## 2. 矩 Moments
- The kth moment of X is $E(X^{k})$, where the first moment is the expectation $\mu$ 
- The kth absolute moment of X is $E(|X|^{k})$. 
- The kth central moment is  $\mu_{k} = E(\{X - E(X)\}^{k}) $$, where $$\mu_{2}$ is the variance of X.
- The skewness coefficient of X is $S_{k}(X) = \frac{\mu_{3}}{(\mu_{2})^{3/2}}$
- The kurtosis of X is $Kur(X) = \frac{\mu_{4}}{(\mu_{2})^{2}}$

### 2.1 What Is Skewness?
Skewness is the degree of asymmetry observed in a probability distribution. When data points on a bell curve are not distributed symmetrically to the left and right sides of the median, the bell curve is skewed. Distributions can be positive and right-skewed ($S_{k} > 0$), or negative and left-skewed ($S_{k} < 0$), as the graph below shows. A normal distribution exhibits zero skewness.
![image](https://github.com/QuantResearchClub/DSA5205/assets/34060865/3f87ce23-14bd-49a1-96e3-5c4dfeb0b25b)

This formula computes skewness: 
$$S_{k}(Y) = E({\frac{Y-E(Y)}{\sigma}})^{3}=E({\frac{(Y-E(Y))^{3}}{\sigma^{3}}})=\frac{E(Y^{3}) - 3\mu\sigma^{2} - \mu^{3}}{\sigma^{3}}$$ 
If Y is a continuous variable, we solve it by integral, otherwise summing all cases in the sample space.
$$S_{k}(Y) = \int_{-\inf}^{\inf} \frac{(Y-E(Y))^{3}}{\sigma^{3}} f(x) dx $$
$$S_{k}(Y) = \sum_{y}  \frac{(Y-E(Y))^{3}}{\sigma^{3}} p(x)$$
Example.
![image](https://github.com/QuantResearchClub/DSA5205/assets/34060865/1cf268de-aae2-4a6c-abf8-74c3aca34ebf)

https://en.wikipedia.org/wiki/Skewness
### 2.2 What is Kurtosis?
Kurtosis is the fourth standardized moment, defined as
$$Kur(Y) = E({\frac{Y-E(Y)}{\sigma}})^{4}=E({\frac{(Y-E(Y))^{4}}{\sigma^{4}}}) = \frac{\mu_{4}}{\sigma^{4}}$$
