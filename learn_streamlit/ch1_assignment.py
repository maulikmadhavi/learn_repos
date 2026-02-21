# ----------- Importing Libraries -----------
import streamlit as st


# --------- Streamlit components -----------
programming_language = st.selectbox(
    "Select your favorite programming langauge",
    ["C++", "JAVA", "Python","Bash"]
)

st.write(f"You have selected: {programming_language} as your favorite programming language. Good choice!")

st.success("Selection done")
