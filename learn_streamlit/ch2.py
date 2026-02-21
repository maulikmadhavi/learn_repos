import streamlit as st

# What is widgets !
# => Elements that are the part of user-interaction: button, select box, input, check box, etc


st.header("Learn Widget and inputs!")

# Use session state to persist button state across reruns
if "chai_ordered" not in st.session_state:
    st.session_state.chai_ordered = False

if st.button("Make Chai"):
    st.session_state.chai_ordered = True

if st.session_state.chai_ordered:
    st.success("Your order registered!")
    st.write("Enter your selection")

    masala = st.checkbox("Masala")
    sugar = st.checkbox("sugar")
    milk = st.checkbox("milk")
    
    # Build options from selected items
    options = ", ".join([opt for opt, selected in [("masala", masala), ("sugar", sugar), ("milk", milk)] if selected]) or "no options"
    
    if st.button("Brew Chai"):
        st.write(f"You chai is being brewed with these options: {options}")
