# ----------- Importing Libraries -----------
import streamlit as st


# ----------- Streamlit App -----------
st.title("Hello User!")
st.header("Welcome to this page!")
st.subheader("This is a subheader")


chai = st.selectbox("Your favorite chai is: ", ["ginger", "masala", "normal"])  # Selectbox for chai options  (message, [values])
# Generic text display
st.text(f"This is a simple text display. You have selected your favorite chai = {chai}")

# Generic write display
st.write(f"This is a write display. You have selected your :red[favorite chai = {chai}]")

# Markdown display
st.markdown(f"This is a markdown display. You have selected your **favorite chai = {chai}**")

st.success("Enjoy your chai! ☕")
