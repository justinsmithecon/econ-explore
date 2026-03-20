"""Thin wrapper for the Human Capital app."""

import streamlit as st

st.set_page_config(page_title="Human Capital", page_icon="📊", layout="wide")

from apps.human_capital.app import main

main()
