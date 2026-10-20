"""
Streamlit dashboard for AI model concentration analysis.

Research question:
Is influential AI model development concentrated in fewer organisations?

Data sources:
1. Stanford AI Index 2025
2. Epoch AI Frontier Models

Visual structure:
1. Long-term notable model leaders, 2014-2024
2. Notable model leaders in 2024
3. Number of organisations participating in frontier model development
4. Top-3 concentration ratio over time
5. Composition of frontier development by organisation
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


# Page setup
st.set_page_config(page_title="AI Model Concentration", page_icon="🤖", layout="wide")

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Data" / "processed"

STANFORD_TOTAL_FILE = (DATA_DIR / "stanford_notable_2014_2024.csv")
STANFORD_2024_FILE = (DATA_DIR / "stanford_notable_2024.csv")

FRONTIER_CONCENTRATION_FILE = (DATA_DIR / "frontier_concentration_by_year.csv")
FRONTIER_YEAR_ORG_FILE = (DATA_DIR / "frontier_by_year_org.csv")

# Load processed data
@st.cache_data
def load_data():
    stanford_total = pd.read_csv(STANFORD_TOTAL_FILE)
    stanford_2024 = pd.read_csv(STANFORD_2024_FILE)
    concentration = pd.read_csv(FRONTIER_CONCENTRATION_FILE)
    frontier_year_org = pd.read_csv(FRONTIER_YEAR_ORG_FILE)

    return (
        stanford_total,
        stanford_2024,
        concentration,
        frontier_year_org)

(stanford_total, stanford_2024, concentration, frontier_year_org) = load_data()


# Header
st.title("Is influential AI model development concentrated in fewer organisations?")
st.markdown("""
    **Target Audience**: Technology policy analyst """)


# Key metrics
latest_year = int(concentration["Year"].max())
latest = concentration[concentration["Year"] == latest_year].iloc[0]
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(f"Frontier organisations in {latest_year}", int(latest["Organisation_count"]))

with col2:
    st.metric(f"Top-3 share in {latest_year}", f'{latest["CR3_percent"]:.1f}%')

with col3:
    st.metric(
        f"Frontier models in {latest_year}",
        int(latest["Frontier_model_count"]))

st.divider()


# Chart 1 - Long-term notable model leaders
st.subheader("1. Who dominated notable AI model development over the past decade?")

st.caption("Number of notable AI models by organisation, 2014-2024.")

long_term = (stanford_total
             .sort_values("Notable_model_count_2014_2024", ascending=True))

fig1 = px.bar(long_term,
              x="Notable_model_count_2014_2024",
              y="Organisation",
              orientation="h",
              text="Notable_model_count_2014_2024",
              labels={
                  "Notable_model_count_2014_2024": "Number of notable AI models",
                  "Organisation": "Organisation"})

fig1.update_traces(textposition="outside", cliponaxis=False)

fig1.update_layout(
    height=430, showlegend=False,
    margin=dict(l=10, r=60, t=20, b=20),
    xaxis_title="Number of notable AI models", yaxis_title="")

st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
    **Interpretation**: A small number of organisations produced most notable AI models over 2014-2024.
    """)
st.divider()


# Chart 2 - 2024 notable model leaders
st.subheader("2. Is current leadership the same?")
st.caption("Number of notable AI models by organisation in 2024.")

recent = (stanford_2024
          .sort_values( "Notable_model_count_2024", ascending=True))

fig2 = px.bar(recent,
              x="Notable_model_count_2024",
              y="Organisation",
              orientation="h",
              text="Notable_model_count_2024",
              labels={
                  "Notable_model_count_2024": "Number of notable AI models",
                  "Organisation": "Organisation"})

fig2.update_traces(textposition="outside", cliponaxis=False)

fig2.update_layout(
    height=430, showlegend=False,
    margin=dict(l=10, r=50, t=20, b=20),
    xaxis_title="Number of notable AI models", yaxis_title="")

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
    **Interpretation:** In 2024, leadership was more competitive, with several organisations producing multiple notable models.
    """)
st.divider()


# Frontier year filter
st.subheader("3. Are fewer organisations participating at the frontier?")

year_min = int(concentration["Year"].min())
year_max = int(concentration["Year"].max())

selected_years = st.slider(
    "Select year range", min_value=year_min,
    max_value=year_max, value=(year_min, year_max))

filtered_concentration = concentration[
    concentration["Year"].between(
        selected_years[0], selected_years[1])].copy()


# Chart 3 - Organisation count over time
fig3 = px.line(
    filtered_concentration,
    x="Year", y="Organisation_count",
    markers=True, text="Organisation_count",
    labels={
        "Year": "Year",
        "Organisation_count": "Number of organisations"})

fig3.update_traces(textposition="top center")

fig3.update_layout(
    height=360, margin=dict(l=10, r=20, t=20, b=20),
    xaxis_title="Year", yaxis_title="Number of organisations")

st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
    **Interpretation:** The number of organisations at the frontier changes a lot from year to year, 
    rather than steadily falling. """)
st.divider()


# Chart 4 - CR3 over time
st.subheader("4. How concentrated is frontier development when participation is broad?")

# Keep only years with more than 3 participating organisations
cr3_meaningful = filtered_concentration[filtered_concentration["Organisation_count"] > 3].copy()

fig4 = px.line(
    cr3_meaningful, x="Year", y="CR3_percent",
    markers=True, text="CR3_percent",
    labels={
        "Year": "Year",
        "CR3_percent": "Top-3 share (%)"})

fig4.update_traces(
    texttemplate="%{text:.0f}%",
    textposition="top center")

fig4.update_layout(
    height=360, margin=dict(l=10, r=20, t=20, b=20),
    xaxis_title="Year", yaxis_title="Top-3 share (%)")

fig4.update_yaxes(range=[0, 100])
fig4.update_xaxes(dtick=1)

st.plotly_chart(fig4, use_container_width=True)

st.markdown('''
    **Interpretation:** : When more than three organisations participate, 
    the top three still hold a large share of frontier development. ''')
st.divider()


# Chart 5 - Frontier composition by organisation
st.subheader("5. Who actually makes up the frontier?")
st.caption("Annual share of frontier AI model development by organisation.")

top_n = st.selectbox("Show leading organisations",
                     options=[3, 5, 7, 10], index=1)

composition = frontier_year_org[
    frontier_year_org["Year"].between(
        selected_years[0],
        selected_years[1])].copy()


# Identify leading organisations over selected period
leading_orgs = (
    composition.groupby("Organisation")["Fractional_contribution"]
    .sum().sort_values(ascending=False).head(top_n).index)


# Group smaller organisations into Others
composition["Display_org"] = (composition["Organisation"].where(
    composition["Organisation"].isin(leading_orgs), "Others"))


# Re-aggregate shares after grouping Others
composition_grouped = (
    composition.groupby(["Year", "Display_org"], as_index=False)
    .agg(Fractional_contribution=("Fractional_contribution", "sum")))


# Complete all year-category combinations so missing years appear as zero
years = list(range(selected_years[0], selected_years[1] + 1))
categories = sorted(composition_grouped["Display_org"].unique())
full_index = pd.MultiIndex.from_product([years, categories], names=["Year", "Display_org"])

composition_grouped = (composition_grouped
                       .set_index(["Year", "Display_org"])
                       .reindex(full_index, fill_value=0)
                       .reset_index())


# Convert contributions to annual percentage shares
composition_grouped["Year_total"] = (
    composition_grouped
    .groupby("Year")["Fractional_contribution"]
    .transform("sum"))

composition_grouped["Share_percent"] = (
    composition_grouped["Fractional_contribution"]
    / composition_grouped["Year_total"] * 100)

composition_grouped["Share_percent"] = (composition_grouped["Share_percent"].fillna(0))


# 100% stacked bar chart
fig5 = px.bar(
    composition_grouped,
    x="Year", y="Share_percent", color="Display_org",
    labels={
        "Year": "Year",
        "Share_percent": "Share of frontier development (%)",
        "Display_org": "Organisation"})

fig5.update_layout(
    height=430, margin=dict(l=10, r=20, t=20, b=20),
    xaxis_title="Year",
    yaxis_title="Share of frontier development (%)",
    legend_title="Organisation",
    barmode="stack")

fig5.update_yaxes(range=[0, 100])
st.plotly_chart(fig5, use_container_width=True)

st.markdown("""
    **Interpretation:** Frontier development is shared across several organisations, 
    but a few leading organisations often make up most of the total. """)
st.divider()


# Overall answer
st.subheader("What does the data suggest?")
st.markdown("""
    Influential AI model development remains concentrated among a small group
    of organisations, but concentration does not increase steadily over time.

    Instead:
    - the number of frontier organisations fluctuates substantially;
    - some years are highly concentrated because only a few
      organisations are represented. """)


# Methodology
with st.expander("Data and methodology"):
    st.markdown("""
        #### Data sources
        I use two data sources: the Stanford AI Index 2025 and Epoch AI Frontier Models.

        #### Method
        For **collaborative models**, each model is divided equally across its listed
        organisations so that the total contribution of one model remains equal to one.

        **Organisation names** are standardised before analysis, and concentration is measured
        using organisation count, Top-3 share (CR3), and HHI.

        #### Organisation harmonisation
        Historical or alternative organisation names are harmonised
        before aggregation. For example, DeepMind, Google Brain,
        Google Research and Google DeepMind are grouped under Google.

        #### Limitations
        Notable AI models and frontier AI models are related but distinct concepts.
        """)


# Sources
st.caption("Sources: Epoch AI Frontier Models dataset & Stanford AI Index Report 2025.")