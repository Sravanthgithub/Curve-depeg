# Curve-depeg
Predictive modelling of depegging on Cuve.fi.

# Curve and the virtual price

The price of token at the stableswap [Curve.fi](https://curve.fi/) is calculated by the the equation below.

$$  A = \frac{A_{contract}}{2} $$ 

<!-- $$  D_1 = \frac{1}{3 \cdot \sqrt[3]{2}} $$
<!-- D_1 = 1 / (3 * 2 ** (1/3)) -->

<!-- $$  D_2 = 432 A y_1 x_1^{2} + 432 A x_1 y_1^{2} $$ -->

<!-- $$ D_3 = \sqrt{6912 (4 A - 1)^{3} x_1^3 y_1^3 + (432 A x_1^2 y_1 + 432 A x_1 y_1^2)^{2}} $$ -->

<!-- $$ D_4 = 4 \sqrt[3]{2} (4 A - 1) x_1 y_1 $$ --> -->

$$ D = \frac{1}{3 \cdot \sqrt[3]{2}} \frac{(432 A y_1 x_1^{2} + 432 A x_1 y_1^{2} + \sqrt{6912 (4 A - 1)^{3} x_1^3 y_1^3 + (432 A x_1^2 y_1 + 432 A x_1 y_1^2)^{2}})^{\frac{1}{3}} - 4 \sqrt[3]{2} (4 A - 1) x_1 y_1} {\sqrt[3]{432 A y_1 x_1^{2} + 432 A x_1 y_1^{2} + \sqrt{6912 (4 A - 1)^{3} x_1^3 y_1^3 + (432 A x_1^2 y_1 + 432 A x_1 y_1^2)^{2}}}}$$

$$ s = \frac{D}{2}$$

<!-- $$ D = D_1 \sqrt[3]{D_2 + D_3} - D_4 / (\sqrt[3]{D_2 + D_3}) $$ -->
$$ y(x) = -\frac{x}{2} - \frac{s}{4A} + s + \frac{\sqrt{(2Ax^2 + sx - 4Asx)^2 + 8Axs^s}}{4Ax}$$

This equation describes a curve whose shape depends on the amount of token ( $tok_1$ and $tok_2$) in the pool and a parameter, A. An example of such a curve is shown in the figure below. The example is of a hypothetical pool with $4 \cdot 10^6 tok_1$, $16 \cdot 10^6 tok_2$ and A = 16.

![The curve](https://github.com/knasterk/Curve-depeg/blob/main/fig/single_curve.png "The curve and the virtual price")
<!-- <img src="[https://github.com/knasterk/Curve-depeg/blob/main/fig/single_curve.png]" width="100" /> -->

![Token ratio and A](https://github.com/knasterk/Curve-depeg/blob/main/fig/curves_A-tokRatio.png "The effect of token ratio and the A parameter")
<!-- <img src="[https://github.com/knasterk/Curve-depeg/blob/main/fig/curves_A-tokRatio.png]" width="100" /> -->


![The region of stable price](https://github.com/knasterk/Curve-depeg/blob/main/fig/A-tokRatio_vprice.png "Higher A leads to a wider region of stable prices but a sharper drop-off")
<!-- <img src="[https://github.com/knasterk/Curve-depeg/blob/main/fig/A-tokRatio_vprice.png]" width="200" /> -->

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
