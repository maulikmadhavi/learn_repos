import streamlit as st

st.title("Hello Chai App!")
st.subheader("Brewed with Streamlit")
st.text("Welcome to your first Streamlit app!")
st.write("Choose your favorite chai from the options below:")
x = st.selectbox("Select your chai:", ["ginger", "masala", "normal"])
st.write(f"You selected: :[{x} chai]. Excellent choice!")

st.success("Enjoy your chai! ☕")