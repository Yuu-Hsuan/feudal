## 1. Manager Loss (經理損失)
$$\cos(\theta) = \frac{\mathbf{S}_{\text{DIFF}} \cdot \mathbf{goal}}{\|\mathbf{S}_{\text{DIFF}}\| \cdot \|\mathbf{goal}\|}$$

$$\text{manager\_loss} = -\mathbb{E} \left[ \text{ADV}_M \cdot \cos(\theta) \right]$$
