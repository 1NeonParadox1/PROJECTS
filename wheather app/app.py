import streamlit as st
from api import get_current_weather

st.set_page_config(
    page_title="Weather Dashboard",
    page_icon="🌤",
    layout="wide"
)

st.title("🌤 Weather Dashboard")

city = st.text_input(
    "Enter City Name",
    "Delhi"
)

if st.button("Get Weather"):

    data = get_current_weather(city)

    if data is None:
        st.error("City not found")
        st.stop()

    current = data["current"]

    temp = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]
    wind = current["wind_speed_10m"]
    pressure = current["surface_pressure"]
    visibility = current["visibility"] / 1000
    uv = current["uv_index"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🌡 Temperature",
        f"{temp} °C"
    )

    col2.metric(
        "💧 Humidity",
        f"{humidity}%"
    )

    col3.metric(
        "🌬 Wind Speed",
        f"{wind} km/h"
    )

    st.divider()

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "🧭 Pressure",
        f"{pressure} mb"
    )

    col5.metric(
        "👁 Visibility",
        f"{visibility:.1f} km"
    )

    col6.metric(
        "☀ UV Index",
        f"{uv}"
    )

    st.success("Weather data fetched successfully!")