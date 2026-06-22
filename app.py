import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_clean_data
from src.tournament import run_tournament
from src.flowchart import analyze_dataset_stats, recommend_technique, plot_flowchart


st.set_page_config(page_title="Extreme Imbalance Stress Lab", page_icon="⚡", layout="wide")

st.title("⚡ Extreme Imbalance Stress Lab")
st.caption("6 techniques. 1 dataset. 99:1 odds. Only one wins.")

st.markdown("""
This dashboard runs 6 imbalance-handling techniques head-to-head on the
KDD Cup 1999 intrusion detection dataset and recommends the best
technique based on YOUR dataset's actual characteristics.
""")

# ── SIDEBAR ──
st.sidebar.header("Controls")
run_button = st.sidebar.button("🏁 Run Tournament", type="primary")

if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "stats" not in st.session_state:
    st.session_state.stats = None

if run_button:
    with st.spinner("Loading data and running tournament... this can take a minute."):
        try:
            results_df = run_tournament()
            X, y = get_clean_data()
            stats = analyze_dataset_stats(X, y)

            st.session_state.results_df = results_df
            st.session_state.stats = stats
            st.success("Tournament complete!")
        except Exception as e:
            # DEFENSIVE: never let the app crash silently on the user.
            # Show a clear error message instead of a blank screen.
            st.error(f"Something went wrong: {e}")

# ── RESULTS DISPLAY ──
if st.session_state.results_df is not None:
    st.subheader("🏆 Tournament Results")
    st.dataframe(
        st.session_state.results_df,
        use_container_width=True
    )

    st.subheader("📊 F1 Score Comparison")
    chart_data = st.session_state.results_df.dropna(subset=["f1"]).set_index("technique")["f1"]
    st.bar_chart(chart_data)

    st.subheader("🧭 Recommendation")
    technique, reason = recommend_technique(
        st.session_state.stats, st.session_state.results_df
    )
    st.markdown(f"**Recommended technique:** `{technique}`")
    st.markdown(f"**Why:** {reason}")

    if st.button("Generate Flowchart Image"):
        plot_flowchart(st.session_state.stats, technique, reason)
        st.image("outputs/flowchart.png")
else:
    st.info("Click 'Run Tournament' in the sidebar to begin.")