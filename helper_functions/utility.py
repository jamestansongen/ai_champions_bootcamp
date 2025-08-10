# Import relevant libraries
import streamlit as st
import hmac

# Function for checking if user has entered password and remember it even as you switch pages
def check_password():
    """Returns True if the user has the correct password stored in secrets.toml"""
    
    def password_entered():
        if hmac.compare_digest(st.session_state.get("password", ""), st.secrets["password"]):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    # Check if user has already entered the correct password
    if st.session_state.get("password_correct", False):
        return True

    # Show password input if not correct yet
    st.text_input("Enter Password:", type="password", key="password", on_change=password_entered)
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("❌ Incorrect password. Please try again.")
    
    return False
