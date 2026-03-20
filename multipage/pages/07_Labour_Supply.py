"""Thin wrapper for the Labour Supply app."""

import streamlit as st

st.set_page_config(page_title="Labour Supply", page_icon="📊", layout="wide")

from apps.labour_supply.app import main

main()
