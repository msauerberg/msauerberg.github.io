---
layout: post
title: "Wasserstein distance and life expectancy differences"
date: 2025-08-09
description: Compare the Wasserstein distance with the difference in life expectancy
tags: [python]
---

# Jupyter Notebook

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