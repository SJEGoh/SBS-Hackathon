import streamlit as st
import pandas as pd

df = pd.DataFrame({
    "Bus": ["5-12", "329-07", "324-03"],
    "DriftPct": [18, 12, 2],
    "Status": ["Review ⚠️⚠️", "Monitor ⚠️", "Normal ✅"]
})

df1 = pd.DataFrame({
    ""
})

STATUS_COLOURS = {
    "Review ⚠️⚠️":  "red",   # light red
    "Monitor ⚠️": "yellow",   # light amber
    "Normal ✅":  "green",   # light green
}

def main():
    st.title("Fleet Risk Warning")
    st.set_page_config(page_title = "Early Warning System")

    @st.dialog("Bus Details")
    def modal(license_plate):
        st.write(f"Details for {license_plate}")
        st.write("Recommendation: [Placeholder]")
        st.write("Bus Details: ")
        st.plotly_chart()
        # Add graph here
    status = st.selectbox("Status",
                          ["Review ⚠️⚠️", "Monitor ⚠️", "Normal ✅"],
                          index = None,
                          placeholder = None)
    license_plate = st.text_input(label = "License Plate",
                                  type = "default",
    )
    
    for i, row in df.iterrows():
        bg = STATUS_COLOURS.get(row["Status"], "#ffffff")

        st.markdown(
            f"""
            <div style="
                background-color: {bg};
                padding: 0.6rem;
                border-radius: 0.4rem;
                margin-bottom: 0.3rem;
            ">
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        c1.write(row["Bus"])
        c2.write(f"{row['DriftPct']}%")
        c3.write(row["Status"])
        if c4.button("Inspect", key=f"inspect_{i}"):
            modal(row["Bus"])
        st.markdown("</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
