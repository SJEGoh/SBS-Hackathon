import streamlit as st

def main():
    st.title("Early Warning System")
    st.set_page_config(page_title = "")

    @st.dialog("Bus Details")
    def modal(license_plate):
        st.write(f"Details for {license_plate}")
        st.write("Recommendation: [Placeholder]")
        # Add graph here
    
    if st.button("SBS8413R"):
        modal("SBS8413R")
        # pull up details for that bus

if __name__ == "__main__":
    main()
