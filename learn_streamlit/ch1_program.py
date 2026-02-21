import streamlit as st

st.title("Hello Prrogramming Selection App")

st.subheader("Choose Your Favorite Programming Language")

programming_languages = [
    "Python",
    "JavaScript",
    "Java",
    "C++",
    "Ruby",
    "Go",
    "Swift",
]
select_language = st.selectbox("Select your programming language:", programming_languages)

st.write(f"You selected: [{select_language}]. Great choice!")

st.success(f"Happy Coding {select_language}! 💻")