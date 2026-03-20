"""Expandable educational content sections for power explorer."""

import streamlit as st


def render_education():
    """Render the 'Learn More' educational sections."""
    st.markdown("---")
    st.subheader("Learn More")

    with st.expander("What is statistical power?"):
        st.markdown("""
**Statistical power** is the probability that a test correctly rejects a false null hypothesis.
In other words, it's the chance of detecting a real effect when one exists.

Power depends on four things:
1. **Effect size** -- how big the true difference is
2. **Sample size** -- more data means more power
3. **Significance level (α)** -- a more lenient threshold gives more power (but more false positives)
4. **Variability** -- less noise makes the signal easier to detect

A common target is **80% power**, meaning you have an 80% chance of detecting the effect if it's real.
""")

    with st.expander("How is analytic power computed?"):
        st.markdown("""
Analytic power uses mathematical formulas based on the **non-central distribution** of the test statistic
under the alternative hypothesis.

For t-tests, this means:
1. Compute the **non-centrality parameter (NCP)** -- this captures how far the true value is from the null
2. Find the **critical value** from the central t-distribution at your α level
3. Compute the probability that a non-central t-distribution exceeds that critical value

This gives an exact answer (given the assumptions), with no randomness involved.
Use the **Analytic** method in the sidebar to see this approach.
""")

    with st.expander("How does simulation estimate power?"):
        st.markdown("""
Monte Carlo simulation estimates power by **repeating the experiment many times**:

1. Generate random data from the **true** distribution (the "real" scenario)
2. Run the statistical test and record whether it rejects H₀
3. Repeat thousands of times
4. **Power ≈ fraction of simulations that rejected H₀**

This approach:
- Makes **no distributional assumptions** beyond the data-generating process
- Gets more precise with more simulations
- Is especially useful when analytic formulas don't exist

Switch to the **Simulation** method in the sidebar to see this approach.
""")
