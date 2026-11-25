from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="House Price Prediction", layout="wide")

MODEL_PATH = Path(__file__).parent / "house_price_model.pkl"

FEATURES = [
    ("MedInc", "Median Income", "%.4f"),
    ("HouseAge", "House Age", "%.1f"),
    ("AveRooms", "Average Rooms", "%.6f"),
    ("AveBedrms", "Average Bedrooms", "%.6f"),
    ("Population", "Population", "%.1f"),
    ("AveOccup", "Average Occupancy", "%.6f"),
    ("Latitude", "Latitude", "%.2f"),
    ("Longitude", "Longitude", "%.2f"),
]

SAMPLE_ROWS = [
    (1.9167, 52.0, 4.262903225806451, 1.0096774193548388, 1212.0, 1.9548387096774194, 37.85, -122.26, 1.592),
    (2.125, 50.0, 4.242424242424242, 1.071969696969697, 697.0, 2.640151515151515, 37.85, -122.26, 1.4),
    (2.775, 52.0, 5.9395770392749245, 1.0483383685800605, 793.0, 2.395770392749245, 37.85, -122.27, 1.525),
    (2.1202, 52.0, 4.052805280528053, 0.966996699669967, 648.0, 2.1386138613861387, 37.85, -122.27, 1.555),
    (1.9911, 50.0, 5.343675417661098, 1.0859188544152745, 990.0, 2.3627684964200477, 37.84, -122.26, 1.587),
    (2.6033, 52.0, 5.465454545454546, 1.0836363636363637, 690.0, 2.5090909090909093, 37.84, -122.27, 1.629),
    (1.3578, 40.0, 4.524096385542169, 1.108433734939759, 409.0, 2.463855421686747, 37.85, -122.27, 1.475),
]

PRESETS = {
    "Default (Custom)": {
        "values": [8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5555, 37.88, -122.23],
        "actual": None,
    },
    **{
        f"Sample {idx} (MedInc: {row[0]:.4f}, Age: {row[1]:.0f}, Pop: {row[4]:.0f}, Actual: ${row[8] * 100000:,.0f})": {
            "values": list(row[:8]),
            "actual": row[8],
        }
        for idx, row in enumerate(SAMPLE_ROWS, start=1)
    },
}


@st.cache_resource
def load_model():
    """Load and cache the trained house price model."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_bytes():
    """Read raw bytes of the trained model file for download."""
    return MODEL_PATH.read_bytes()


def apply_preset():
    """Populate input fields in session state from the selected preset."""
    selected = PRESETS[st.session_state["preset_choice"]]
    for (key, _, _), val in zip(FEATURES, selected["values"]):
        st.session_state[key] = float(val)


def prediction_page():
    """Render the house price prediction form page."""
    model = load_model()

    if "preset_choice" not in st.session_state:
        st.session_state["preset_choice"] = "Default (Custom)"
        apply_preset()

    st.title("House Price Prediction")

    st.selectbox(
        "Choose Sample Data",
        options=list(PRESETS.keys()),
        key="preset_choice",
        on_change=apply_preset,
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        for idx, (key, label, fmt) in enumerate(FEATURES):
            target_col = col1 if idx < 4 else col2
            with target_col:
                st.number_input(label, key=key, format=fmt)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_values = [st.session_state[key] for key, _, _ in FEATURES]
        prediction = model.predict([input_values])[0]
        st.success(f"Predicted House Price: ${prediction * 100000:,.2f}")

        preset = PRESETS[st.session_state["preset_choice"]]
        if preset["actual"] is not None and all(
            abs(val - preset_val) < 1e-6 for val, preset_val in zip(input_values, preset["values"])
        ):
            st.info(f"Actual House Price: ${preset['actual'] * 100000:,.2f}")


pages = [
    st.Page(prediction_page, title="Predict Price", icon=":material/home:", default=True),
    st.Page(str(Path(__file__).parent / "pages" / "dataset.py"), title="View & Download Dataset", icon=":material/table_chart:"),
]

pg = st.navigation(pages)

with st.sidebar:
    st.divider()
    model_bytes = load_model_bytes()
    st.caption(f"Model: `house_price_model.pkl` ({len(model_bytes) / 1024:.2f} KB)")
    st.download_button(
        label="Download Model (.pkl)",
        data=model_bytes,
        file_name="house_price_model.pkl",
        mime="application/octet-stream",
        width="stretch",
    )

pg.run()
