"""Thin wrapper for the Central Limit Theorem Demo app."""

import streamlit as st

st.set_page_config(page_title="CLT Demo", page_icon="📊", layout="wide")

from apps.clt_demo.app import main

main()
