"""Thin wrapper for the Spence Signalling app."""

import streamlit as st

st.set_page_config(page_title="Signalling", page_icon="📊", layout="wide")

from apps.signalling.app import main

main()
