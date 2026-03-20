"""Economics & Econometrics Interactive Explorer — Multi-page hub."""

import streamlit as st

st.set_page_config(
    page_title="Econ Explorer",
    page_icon="📊",
    layout="wide",
)

st.title("Economics & Econometrics Interactive Explorer")

st.markdown("""
Welcome to the **Econ Explorer** — a collection of interactive apps for learning
economics and econometrics concepts visually.

Use the sidebar to navigate between apps.

---

### Available Apps

| App | Category | Description |
|-----|----------|-------------|
| **Statistical Power Explorer** | Core Statistics | How effect size, sample size, variance, and alpha affect power |
| **Central Limit Theorem Demo** | Core Statistics | Draw samples from non-normal populations, watch x̄ converge to normal |
| **Oaxaca-Blinder Decomposition** | Labour Economics | Decompose wage gaps into explained vs unexplained components |
| **Tax Incidence** | Public Economics | How tax burden splits between buyers/sellers depending on elasticities |
| **Human Capital** | Labour Economics | Mincer equation, returns to schooling, and ability bias |
| **Labour Demand** | Labour Economics | Substitution and scale effects of a wage change |

---

### Coming Soon

- Confidence Interval Coverage
- Law of Large Numbers
- OLS Visualizer
- Omitted Variable Bias
- Supply and Demand
- And many more...
""")
