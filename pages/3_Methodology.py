import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="Methodology",
    layout="centered"
)

# Page title
st.title("Methodology")

# Path to the PNG file
png_path = os.path.join(os.path.dirname(__file__), "..", "flowchart.png")

# Display flowchart if exists
if os.path.exists(png_path):
    st.image(png_path, caption="DocuMind flowchart", use_container_width=True)
else:
    st.warning(f"Flowchart not found at: {png_path}")
