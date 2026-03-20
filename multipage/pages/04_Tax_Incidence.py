"""Thin wrapper for the Tax Incidence app."""

import streamlit as st

st.set_page_config(page_title="Tax Incidence", page_icon="📊", layout="wide")

from apps.tax_incidence.app import main

main()
