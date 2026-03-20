"""Thin wrapper for the Oaxaca-Blinder Decomposition app."""

import streamlit as st

st.set_page_config(page_title="Oaxaca-Blinder", page_icon="📊", layout="wide")

from apps.oaxaca_blinder.app import main

main()
