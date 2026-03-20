"""Thin wrapper for the Statistical Power Explorer app."""

import streamlit as st

st.set_page_config(page_title="Power Explorer", page_icon="📊", layout="wide")

from apps.power_explorer.app import main

main()
