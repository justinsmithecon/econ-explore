"""Central Limit Theorem Demo — Streamlit app.

Draw samples from non-normal populations and watch the sampling distribution
of the mean converge to a normal distribution as sample size grows.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import streamlit as st

from shared.base import SliderSpec, SelectSpec, SamplingConcept


# ---------------------------------------------------------------------------
# Population distributions
# ---------------------------------------------------------------------------

POPULATIONS = {
    "Exponential (λ=1)": {
        "rvs": lambda rng, n: rng.exponential(1.0, size=n),
        "mean": 1.0,
        "std": 1.0,
        "pdf_x": np.linspace(0, 6, 300),
        "pdf_y": stats.expon.pdf(np.linspace(0, 6, 300)),
    },
    "Uniform (0, 1)": {
        "rvs": lambda rng, n: rng.uniform(0, 1, size=n),
        "mean": 0.5,
        "std": 1 / np.sqrt(12),
        "pdf_x": np.linspace(-0.2, 1.2, 300),
        "pdf_y": stats.uniform.pdf(np.linspace(-0.2, 1.2, 300)),
    },
    "Bernoulli (p=0.3)": {
        "rvs": lambda rng, n: rng.binomial(1, 0.3, size=n).astype(float),
        "mean": 0.3,
        "std": np.sqrt(0.3 * 0.7),
        "pdf_x": np.array([0, 1]),
        "pdf_y": np.array([0.7, 0.3]),
    },
    "Chi-squared (df=3)": {
        "rvs": lambda rng, n: rng.chisquare(3, size=n),
        "mean": 3.0,
        "std": np.sqrt(6.0),
        "pdf_x": np.linspace(0, 15, 300),
        "pdf_y": stats.chi2.pdf(np.linspace(0, 15, 300), 3),
    },
    "Bimodal (mixture)": {
        "rvs": lambda rng, n: np.where(
            rng.uniform(size=n) < 0.5,
            rng.normal(-2, 0.8, size=n),
            rng.normal(2, 0.8, size=n),
        ),
        "mean": 0.0,
        "std": np.sqrt(4 + 0.64),  # mixture variance
        "pdf_x": np.linspace(-5, 5, 300),
        "pdf_y": 0.5 * stats.norm.pdf(np.linspace(-5, 5, 300), -2, 0.8)
               + 0.5 * stats.norm.pdf(np.linspace(-5, 5, 300), 2, 0.8),
    },
}


class CLTDemo(SamplingConcept):
    @property
    def name(self) -> str:
        return "Central Limit Theorem Demo"

    @property
    def description(self) -> str:
        return (
            "The Central Limit Theorem states that the sampling distribution of "
            "the sample mean approaches a normal distribution as sample size increases, "
            "regardless of the population's shape. Explore this by drawing samples "
            "from various non-normal populations."
        )

    def params(self) -> list:
        return [
            SelectSpec("population", "Population distribution",
                       list(POPULATIONS.keys()), help_text="Choose the shape of the underlying population"),
            SliderSpec("n", "Sample size (n)", 1.0, 200.0, 5.0, 1.0,
                       "Number of observations per sample"),
            SliderSpec("n_samples", "Number of samples", 100.0, 10000.0, 1000.0, 100.0,
                       "How many sample means to draw"),
        ]

    def population_distribution(self, params):
        return POPULATIONS[params["population"]]

    def draw_samples(self, params, rng):
        pop = POPULATIONS[params["population"]]
        n = int(params["n"])
        n_samples = int(params["n_samples"])
        means = np.array([pop["rvs"](rng, n).mean() for _ in range(n_samples)])
        return {"means": means, "pop_mean": pop["mean"], "pop_std": pop["std"]}

    def render(self, params, depth="undergraduate"):
        pass

    def educational_sections(self, depth="undergraduate"):
        sections = [
            ("What is the Central Limit Theorem?",
             "The **Central Limit Theorem (CLT)** says that if you take many "
             "random samples of size *n* from *any* population with finite mean μ "
             "and finite variance σ², then the distribution of the sample means "
             "x̄ approaches a normal distribution:\n\n"
             r"$$\bar{x} \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$"
             "\n\nThis holds regardless of the population's original shape. "
             "The convergence is faster for populations that are already symmetric."),
            ("Why does the CLT matter?",
             "The CLT is the foundation for most of inferential statistics:\n\n"
             "- It justifies using **z-tests** and **t-tests** even when the data isn't normal\n"
             "- It explains why **confidence intervals** work\n"
             "- It's why the normal distribution appears so often in practice\n\n"
             "Without the CLT, we'd need to know the exact population distribution "
             "to do hypothesis testing — which is rarely possible."),
        ]
        if depth == "graduate":
            sections.append((
                "Berry-Esseen bound",
                "The **Berry-Esseen theorem** quantifies the rate of convergence:\n\n"
                r"$$\sup_x \left| P\left(\frac{\bar{x} - \mu}{\sigma/\sqrt{n}} \le x\right) - \Phi(x) \right| "
                r"\le \frac{C \cdot \rho}{\sigma^3 \sqrt{n}}$$"
                "\n\nwhere ρ = E[|X - μ|³] is the third absolute moment and C ≤ 0.4748. "
                "This tells us convergence is O(1/√n) and is slower for skewed distributions."
            ))
        return sections


def main():
    """Main entry point for the CLT Demo app."""
    concept = CLTDemo()

    st.sidebar.header("Central Limit Theorem Demo")
    st.sidebar.markdown("---")

    # Render params
    pop_name = st.sidebar.selectbox("Population distribution", list(POPULATIONS.keys()))
    n = st.sidebar.slider("Sample size (n)", min_value=1, max_value=200, value=5, step=1,
                          help="Number of observations per sample")
    n_samples = st.sidebar.slider("Number of samples", min_value=100, max_value=10000,
                                  value=1000, step=100,
                                  help="How many sample means to draw")
    depth = st.sidebar.radio("Depth", ["undergraduate", "graduate"],
                             format_func=lambda x: x.title())

    use_seed = st.sidebar.checkbox("Fix random seed", value=True)
    seed = 42 if use_seed else None

    params = {"population": pop_name, "n": n, "n_samples": n_samples}
    pop = POPULATIONS[pop_name]
    rng = np.random.default_rng(seed)
    result = concept.draw_samples(params, rng)

    # --- Main area ---
    st.title("Central Limit Theorem Demo")
    st.markdown(concept.description)

    st.markdown("---")

    st.markdown("""
**How this app works:** Pick a population distribution (none of which are
normal) and a sample size *n*. The app repeatedly draws samples of size *n*,
computes each sample mean, and plots the resulting sampling distribution.
As you increase *n*, watch the histogram of sample means converge to a
normal curve — that's the Central Limit Theorem in action.
""")

    st.markdown("---")

    # Two panels: population + sampling distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Population Distribution**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        if pop_name == "Bernoulli (p=0.3)":
            ax1.bar([0, 1], [0.7, 0.3], width=0.3, color="#2563eb", alpha=0.7)
            ax1.set_xticks([0, 1])
        else:
            ax1.plot(pop["pdf_x"], pop["pdf_y"], color="#2563eb", linewidth=2)
            ax1.fill_between(pop["pdf_x"], pop["pdf_y"], color="#2563eb", alpha=0.15)
        ax1.axvline(pop["mean"], color="#dc2626", linestyle="--", linewidth=1.5,
                    label=f"μ = {pop['mean']:.2f}")
        ax1.set_xlabel("x", fontsize=10)
        ax1.set_ylabel("Density", fontsize=10)
        ax1.set_title(pop_name, fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.markdown(f"**Sampling Distribution of x̄  (n = {n})**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))

        means = result["means"]
        ax2.hist(means, bins=50, density=True, color="#2563eb", alpha=0.6,
                 edgecolor="white", linewidth=0.5, label=f"{n_samples} sample means")

        # Overlay CLT normal
        se = pop["std"] / np.sqrt(n)
        x_norm = np.linspace(pop["mean"] - 4 * se, pop["mean"] + 4 * se, 300)
        ax2.plot(x_norm, stats.norm.pdf(x_norm, pop["mean"], se),
                 color="#dc2626", linewidth=2, label=f"N(μ, σ²/n)")

        ax2.axvline(pop["mean"], color="#64748b", linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_xlabel("Sample mean (x̄)", fontsize=10)
        ax2.set_ylabel("Density", fontsize=10)
        ax2.set_title("Sampling Distribution", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Summary stats
    means = result["means"]
    se = pop["std"] / np.sqrt(n)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Population μ", f"{pop['mean']:.3f}")
    col2.metric("Mean of x̄", f"{means.mean():.3f}")
    col3.metric("Theoretical SE", f"{se:.3f}")
    col4.metric("Observed SD of x̄", f"{means.std():.3f}")

    # Normality test
    if n >= 3 and len(means) >= 20:
        _, shapiro_p = stats.shapiro(means[:5000])  # shapiro limited to 5000
        if shapiro_p > 0.05:
            st.success(f"Shapiro-Wilk test: p = {shapiro_p:.4f} — cannot reject normality")
        else:
            st.warning(f"Shapiro-Wilk test: p = {shapiro_p:.4f} — sampling distribution is not yet normal (try increasing n)")

    st.markdown("---")
    st.markdown(
        "**Key insight:** As you increase the sample size *n*, the histogram on the "
        "right becomes more bell-shaped and clusters more tightly around the true mean, "
        "regardless of the population's shape on the left."
    )

    # Educational sections
    sections = concept.educational_sections(depth)
    if sections:
        st.markdown("---")
        st.subheader("Learn More")
        for title, body in sections:
            with st.expander(title):
                st.markdown(body)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Central Limit Theorem Demo",
        page_icon="📊",
        layout="wide",
    )
    main()
