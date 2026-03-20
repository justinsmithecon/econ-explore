"""Thin wrapper for the Statistical Discrimination app."""

import streamlit as st

st.set_page_config(page_title="Statistical Discrimination", page_icon="📊", layout="wide")

from apps.statistical_discrimination.app import main

main()
