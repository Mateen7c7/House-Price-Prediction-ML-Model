from pathlib import Path

import pandas as pd
import streamlit as st

DATASET_PATH = Path(__file__).resolve().parent.parent / "house_prices.csv"


@st.cache_data
def load_dataset():
    """Load the house prices CSV as a DataFrame and raw bytes for download."""
    df = pd.read_csv(DATASET_PATH)
    csv_bytes = DATASET_PATH.read_bytes()
    return df, csv_bytes


df, csv_bytes = load_dataset()
file_size_kb = len(csv_bytes) / 1024

st.title("House Prices Dataset")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1.5])
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("File Size", f"{file_size_kb:,.1f} KB")
with col4:
    st.download_button(
        label="Download Full CSV",
        data=csv_bytes,
        file_name="house_prices.csv",
        mime="text/csv",
        type="primary",
        width="stretch",
    )

with st.expander("Filter Dataset"):
    selected_columns = st.multiselect(
        "Columns to display",
        options=list(df.columns),
        default=list(df.columns),
    )

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        med_inc_range = st.slider(
            "Median Income (MedInc)",
            float(df["MedInc"].min()),
            float(df["MedInc"].max()),
            (float(df["MedInc"].min()), float(df["MedInc"].max())),
        )
    with fcol2:
        house_age_range = st.slider(
            "House Age (HouseAge)",
            float(df["HouseAge"].min()),
            float(df["HouseAge"].max()),
            (float(df["HouseAge"].min()), float(df["HouseAge"].max())),
        )
    with fcol3:
        price_range = st.slider(
            "Median House Value (MedHouseVal)",
            float(df["MedHouseVal"].min()),
            float(df["MedHouseVal"].max()),
            (float(df["MedHouseVal"].min()), float(df["MedHouseVal"].max())),
        )

filtered_df = df[
    df["MedInc"].between(*med_inc_range)
    & df["HouseAge"].between(*house_age_range)
    & df["MedHouseVal"].between(*price_range)
]

if not selected_columns:
    st.warning("Select at least one column to display.")
else:
    display_df = filtered_df[selected_columns]
    st.caption(f"Showing {len(display_df):,} of {len(df):,} rows")
    st.dataframe(display_df, height=500)

    if len(filtered_df) < len(df) or len(selected_columns) < len(df.columns):
        st.download_button(
            label="Download Filtered CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="house_prices_filtered.csv",
            mime="text/csv",
        )

with st.expander("Summary Statistics"):
    st.dataframe(filtered_df.describe())
