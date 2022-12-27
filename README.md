# Curve-depeg
Predictive modelling of depegging on Cuve.fi.

## Curve and the virtual price

The price of token at the stableswap [Curve.fi](https://curve.fi/) is calculated by the the equation below.

$$  A = \frac{A_{contract}}{2} $$

<!-- $$  D_1 = \frac{1}{3 \cdot \sqrt[3]{2}} $$ -->
<!-- D_1 = 1 / (3 * 2 ** (1/3)) -->

<!-- $$  D_2 = 432 A y_1 x_1^{2} + 432 A x_1 y_1^{2} $$ -->

<!-- $$ D_3 = \sqrt{6912 (4 A - 1)^{3} x_1^3 y_1^3 + (432 A x_1^2 y_1 + 432 A x_1 y_1^2)^{2}} $$ -->

<!-- $$ D_4 = 4 \sqrt[3]{2} (4 A - 1) x_1 y_1 $$ -->

$$ D = \frac{1}{3 \cdot \sqrt[3]{2}} \frac{(432 A y_1 x_1^{2} + 432 A x_1 y_1^{2} + \sqrt{6912 (4 A - 1)^{3} x_1^3 y_1^3 + (432 A x_1^2 y_1 + 432 A x_1 y_1^2)^{2}})^{\frac{1}{3}} - 4 \sqrt[3]{2} (4 A - 1) x_1 y_1} {\sqrt[3]{432 A y_1 x_1^{2} + 432 A x_1 y_1^{2} + \sqrt{6912 (4 A - 1)^{3} x_1^3 y_1^3 + (432 A x_1^2 y_1 + 432 A x_1 y_1^2)^{2}}}}$$

$$ s = \frac{D}{2}$$

<!-- $$ D = D_1 \sqrt[3]{D_2 + D_3} - D_4 / (\sqrt[3]{D_2 + D_3}) $$ -->
$$ y(x) = -\frac{x}{2} - \frac{s}{4A} + s + \frac{\sqrt{(2Ax^2 + sx - 4Asx)^2 + 8Axs^s}}{4Ax}$$

The equation describes a curve whose shape depends on the amount of token $(x_1$ and $y_1)$ in the pool and a parameter, $A$. An example of such a curve is shown in the figure below. The example is of a hypothetical pool with $4 \cdot 10^6$ of $tok_1$, $16 \cdot 10^6$ of $tok_2$ and $A = 16$.

![The curve](https://github.com/knasterk/Curve-depeg/blob/main/fig/single_curve.png "The curve and the virtual price")
<!-- <img src="[https://github.com/knasterk/Curve-depeg/blob/main/fig/single_curve.png]" width="100" /> -->

The virtual price is given by the negative of the tangent to the point $(x_1, y_1)$ on the curve. In the figure the tangent is shown by dashed lines.

Changing the number of tokens and/or A changes the shape of the curve which in turn, influences the price. The figure below shows examples of this.

![Token ratio and A](https://github.com/knasterk/Curve-depeg/blob/main/fig/curves_A-tokRatio.png "The effect of token ratio and the A parameter")
<!-- <img src="[https://github.com/knasterk/Curve-depeg/blob/main/fig/curves_A-tokRatio.png]" width="100" /> -->

The more extreme the ratio of the two tokens become the more the price deviates from 1. This can be counteracted by increasing A, which leads to a greater range of ratios with a price close to 1, but sharper drop-offs at the end of this range. This is shown over a greater range of token ratios and values for A in the figure below.

![The region of stable price](https://github.com/knasterk/Curve-depeg/blob/main/fig/A-tokRatio_vprice.png "Higher A leads to a wider region of stable prices but a sharper drop-off")
<!-- <img src="[https://github.com/knasterk/Curve-depeg/blob/main/fig/A-tokRatio_vprice.png]" width="200" /> -->

In the figure above, the red line indicates the border between peg ( $0.95 \geq price_{x_1} \leq 1$) and depeg ( $price_{x_1}< 0.95$).

## Predicting depeg

We want to predict depegs ( $price_{x_1} < 0.95$ or $price_{x_1} > 1.05$) 24 hours in advance.

### Data
We have data from five [Curve.fi](https://curve.fi) pools where depeg occurred ([USDN-3CRV](https://curve.fi/usdn), [MIM-UST](https://curve.fi/factory/48), [sETH-ETH](https://curve.fi/seth), [pUSd-3Crv](https://curve.fi/factory/113), [UST-3Pool](https://curve.fi/factory/53)). From each pool we have the number of tokens over variable number of days, from 77 (pUSd-3Crv) to 626 (USDN-3CRV) days. We computed the virtual price from the pool data. We defined a depeg as a 1% deviation from a price of 1, perfect peg. Thresholding the price data gave us binary time series that we trained learners to predict future depegs (24 hours).

### Modelling
Learners were fitted three times on four of the five pools, each time with a different hold-out pool. We averaged the results across the three different runs and selected the learner and window length resulting in the highest $F_1$ score.

We compared six different learners and 19 window lengths between 1 and 70 days.

### Learners
 * Logistic Regression
 * Naive Bayes Classifier
 * Support Vector Classifier
 * Decision Tree Classifier
 * Random Forest Classifier
 * Gradient Boosting Classifier

### Results

We got the best results with a Gradient Boosting Classifier and a window length of three days.

![Predicting depegs](https://github.com/knasterk/Curve-depeg/blob/main/fig/depeg_predictions_thresh-1.0pct.png "Depeg predictions")

![Performance](https://github.com/knasterk/Curve-depeg/blob/main/fig/depeg_confuse_ROC_thresh-1.0pct.png "Performance")

The results are surprisingly good and could be used to manage pools.

## Use
Paths are relative to the working directory.

Search for the best predictive model:
```python
from model import search_cls
best, results = search_cls()
```

Fit, predict and plot:
```python
from model import fit_predict
fit_predict()
```
