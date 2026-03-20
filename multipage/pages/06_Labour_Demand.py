"""Thin wrapper for the Labour Demand Decomposition app."""

import streamlit as st

st.set_page_config(page_title="Labour Demand", page_icon="📊", layout="wide")

from apps.labour_demand.app import main

main()
