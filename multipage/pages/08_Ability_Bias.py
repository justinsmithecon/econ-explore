"""Thin wrapper for the Ability Bias app."""

import streamlit as st

st.set_page_config(page_title="Ability Bias", page_icon="📊", layout="wide")

from apps.ability_bias.app import main

main()
