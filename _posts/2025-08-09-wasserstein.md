---
layout: post
title: "Wasserstein distance and life expectancy differences"
date: 2025-08-09
description: Compare the Wasserstein distance with the difference in life expectancy
tags: [python]
---

# Wasserstein distance and the difference in life expectancy

## Short description of the Wasserstein distance

The Wasserstein distance (or earth mover's distance) computes the minimum "work" needed to transform one distribution into the other and refers to *optimal transport theory*.
According to [https://docs.scipy.org](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html), the Wasserstein distance "is a similarity metric between two probability distributions. In the discrete case, the Wasserstein distance can be understood as the cost of an optimal transport plan to convert one distribution into the other. The cost is calculated as the product of the amount of probability mass being moved and the distance it is being moved."

In the one-dimensional case and when the cost function is defined as $c(x,y)=|x-y|$, the Wasserstein distance can be expressed as,
$$
W(u,v)=\int_{-\infty}^{\infty}|U-V|,
$$
where $U$ and $V$ are the respective cumulative distribution functions (CDF) of $u$ and $v$.
See Santambrogio (2015), chapter 2 for more details [Santambrogio (2015)](https://link.springer.com/book/10.1007/978-3-319-20828-2).

## Short description of life expectancy at birth

In demography, life expectancy at birth ($e_0$) measures the expected number of life years on the basis of a set of age-specific mortality rates. $e_0$ is derived from a period life table - a procedure where age-specific mortality rates are applied to hypothetical cohort to calculate the number of persons alive and deceased in each age interval. This table will then be used to derive the expected number of life years. Usually, the size of the initial life table population is set to 100 000 persons. The number of deceased persons in each age interval is denoted as $d_x$, while $l_x$ gives the age-specific number of persons alive. When setting the initial size of the life table population to $1$, the $d_x$ function can be seen as a probability density function (PDF). $e_0$ is the mean age at death,
$$
e_0=\frac{\int_0^{\omega}x\cdot d(x)dx}{\int_0^{\omega}d(x)dx},
$$
where $\omega$ denotes the upper age interval.
A more popular formular, however, is expressing $e_0$ as the area under the survivorship curve,
$$
e_0=\int_0^{\omega}l(x)dx.
$$
The difference between two $e_0$ values is then,
$$
e_{0,A}-e_{0,A} = \int_0^{\omega}(l_A(x)-l_B(x))dx
$$
## The relationship between the Wasserstein distance and difference between two life expectancy at birth values

It is well-known in surival analysis that PDF, CDF, and the survivorship function are directly linked to each other. That is, the survivorship function can be derived from the PDF and is complement of the CDF,
$$
S(x)=1-\int_{-\infty}^x f(u)du\\
=1-F(x).
$$

As shown above, the Wasserstein distance is the area between two CDFs. Substituting the CDFs with the life table survivorship function $l_x$ gives,
$$
W(d_A,d_B)=\int_{0}^{\omega}|l_A(x)-l_B(x)|dx.
$$
This implies that the difference between two $e_0$ values is equal to the Wasserstein distance between two life table age-at-death distributions, whenever $l_A(x) >= l_B(x)$ for all $x$.

## Demonstration in Python using life tables from the Human Mortality Database

```liquid
{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/wasserstein.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/wasserstein.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
```