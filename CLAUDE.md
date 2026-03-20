# Economics & Econometrics Interactive Explorer

A monorepo of interactive Streamlit apps for teaching economics and econometrics concepts.

## Setup
```bash
pip install -e ".[dev]"
```

## Running individual apps
```bash
streamlit run apps/power_explorer/app.py
streamlit run apps/clt_demo/app.py
```

## Running the multi-page hub
```bash
streamlit run multipage/Home.py
```

## Testing
```bash
python -m pytest tests/
```

## Architecture

### Shared library (`shared/`)
- `base.py` — `InteractiveConcept` ABC + category bases (`HypothesisTestConcept`, `SamplingConcept`, `EquilibriumConcept`, `EstimationConcept`) + `SliderSpec`/`SelectSpec`
- `stats_utils.py` — Analytic power helpers (non-central t, normal approx)
- `simulation.py` — Generic Monte Carlo runner
- `plotting.py` — Reusable matplotlib helpers (distribution plots, shading)
- `ui.py` — Generic Streamlit param renderer, depth toggle

### Apps (`apps/`)
Each app lives in its own directory with an `app.py` containing a `main()` function.
- `power_explorer/` — Statistical power analysis (4 test scenarios)
- `clt_demo/` — Central Limit Theorem visualization
- `oaxaca_blinder/` — Oaxaca-Blinder wage gap decomposition
- `tax_incidence/` — Tax incidence with supply/demand
- `human_capital/` — Returns to schooling investment decision
- `labour_demand/` — Substitution and scale effects of wage changes
- `labour_supply/` — Income and substitution effects on hours worked
- `ability_bias/` — Omitted variable bias in returns to schooling
- `signalling/` — Spence job-market signalling model
- `statistical_discrimination/` — Arrow/Phelps Bayesian wage-setting

### Multi-page hub (`multipage/`)
- `Home.py` — Landing page
- `pages/` — Thin wrappers that import and call each app's `main()`

## Adding a new app
1. Create `apps/<name>/app.py` with a `main()` function
2. Pick the right base class from `shared/base.py`
3. Add a two-line wrapper in `multipage/pages/`
4. Add tests in `tests/test_<name>/`

## App conventions
- Every app must include a **"How this app works"** blurb right after the title and description, separated by horizontal lines. This should explain in plain language what the sliders control (e.g. population parameters vs sample parameters), what the app does with them (e.g. draws a sample, runs estimation), and what the outputs show. Keep it to a short paragraph.
- **Chart layout:** Never put more than two charts side by side — three squeezed columns are too small to read. If an app has three or more charts, use a stacked layout: the most detailed chart gets full width, and simpler/comparable charts go side by side in a two-column row. Put the most important or most detailed chart first.
- **Graphs above numbers:** Visualizations come first, metrics/tables below. The graph is the primary teaching tool; numbers support it.
- **Keep models simple:** Default to fewer variables and simpler specifications for pedagogical clarity. Can always add complexity later.
- **Consistency in frameworks:** If one metric uses discounted values (e.g. NPV), all related metrics in the same section should too. Don't mix nominal and PV in the same analytical framework.
- **Decomposition annotations:** For regression-based decompositions, use points and arrows directly on the regression lines. The counterfactual point sits ON the reference group's line, not on a separate line.
- **Build iteratively:** New apps are built based on interest, not in a fixed order. Ask which app to build next rather than assuming a sequence.
