"""Generic Streamlit UI helpers for rendering parameters and depth toggles."""

from __future__ import annotations

from typing import Any

import streamlit as st

from shared.base import ParamSpec, SliderSpec, SelectSpec


def render_params(param_specs: list[ParamSpec], prefix: str = "") -> dict[str, Any]:
    """Render sidebar widgets from a list of ParamSpec and return current values."""
    params = {}
    for spec in param_specs:
        key = f"{prefix}_{spec.key}" if prefix else spec.key
        if isinstance(spec, SliderSpec):
            val = st.sidebar.slider(
                spec.label,
                min_value=spec.min_value,
                max_value=spec.max_value,
                value=spec.default,
                step=spec.step,
                help=spec.help_text,
                key=key,
            )
        elif isinstance(spec, SelectSpec):
            val = st.sidebar.selectbox(
                spec.label,
                options=spec.options,
                index=spec.options.index(spec.default),
                help=spec.help_text,
                key=key,
            )
        else:
            continue
        params[spec.key] = val
    return params


def render_depth_toggle() -> str:
    """Render a depth toggle in the sidebar. Returns 'undergraduate' or 'graduate'."""
    depth = st.sidebar.radio(
        "Depth",
        ["undergraduate", "graduate"],
        format_func=lambda x: x.title(),
        help="Controls the level of mathematical detail shown",
    )
    return depth


def render_educational_sections(sections: list[tuple[str, str]]):
    """Render expandable educational content sections."""
    if not sections:
        return
    st.markdown("---")
    st.subheader("Learn More")
    for title, body in sections:
        with st.expander(title):
            st.markdown(body)
