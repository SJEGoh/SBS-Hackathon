import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fleet Risk Warning")

# Residual change thresholds
THRESHOLD_A = 0.025  # Monitor threshold
THRESHOLD_B = 0.05  # Review threshold

def get_bus_data():
    """Fetch all trips data from the database grouped by bus"""
    conn = sqlite3.connect('sbs.db')
    query = """
    SELECT b.license_plate, t.expected, t.actual, t.residual, t.datetime
    FROM trips t
    JOIN buses b ON t.bus_id = b.id
    ORDER BY b.license_plate, t.datetime
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def calculate_status(residuals, threshold_a, threshold_b):
    """
    Determine bus status based on change in residuals:
    - "Monitor": Change in residual > threshold_a but < threshold_b
    - "Review": Change in residual > threshold_b
    - "Normal": Otherwise
    """
    if len(residuals) < 2:
        return "Normal", 0.0
    
    residuals = np.array(residuals)
    
    # Calculate the maximum change in residual across consecutive trips
    residual_changes = np.abs(np.diff(residuals))
    max_change = np.max(residual_changes)
    
    if max_change > threshold_b:
        return "Review", max_change
    elif max_change > threshold_a:
        return "Monitor", max_change
    else:
        return "Normal", max_change

def get_status_display(status):
    """Format status with emoji and color"""
    if status == "Review":
        return "Review ⚠️⚠️", "red"
    elif status == "Monitor":
        return "Monitor ⚠️", "yellow"
    else:
        return "Normal ✅", "green"

def create_bus_plot(bus_data):
    """Create a plot of expected vs actual fuel efficiency"""
    fig = go.Figure()
    
    # Add expected values
    fig.add_trace(go.Scatter(
        x=list(range(len(bus_data))),
        y=bus_data['expected'],
        mode='lines+markers',
        name='Expected',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=list(range(len(bus_data))),
        y=bus_data['actual'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"Fuel Efficiency Trend",
        xaxis_title="Trip Number",
        yaxis_title="Fuel Efficiency (KML)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def main():
    st.title("Fleet Risk Warning System")
    
    curr_status = st.selectbox("Status",
                          ["Review ⚠️⚠️", "Monitor ⚠️", "Normal ✅"],
                          index = None,
                          placeholder = None)
    license_plate = st.text_input(label = "License Plate",
                                  type = "default",
    )
    # Fetch data
    df_trips = get_bus_data()
    
    if df_trips.empty:
        st.warning("No trip data found in database. Please run main.py to generate data.")
        return
    
    # Group by bus
    buses = df_trips['license_plate'].unique()
    bus_status_list = []
    
    for bus in buses:
        bus_data = df_trips[df_trips['license_plate'] == bus].reset_index(drop=True)
        status, max_change = calculate_status(bus_data['residual'].values, THRESHOLD_A, THRESHOLD_B)
        bus_status_list.append({
            'Bus': bus,
            'Status': status,
            'Trip Count': len(bus_data),
            'Max Residual Change': max_change
        })
    
    # Display bus status cards
    st.subheader("Bus Fleet Status Overview")
    @st.dialog("Bus Details", width = "medium")
    def modal(license_plate):
        bus_data = df_trips[df_trips['license_plate'] == license_plate].reset_index(drop=True)
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            fig = create_bus_plot(bus_data)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Total Trips", len(bus_data))
            # Calculate residual changes
            residuals = bus_data['residual'].values
            if len(residuals) > 1:
                residual_changes = np.abs(np.diff(residuals))
                st.metric("Max Residual Change", f"{np.max(residual_changes):.3f}")
                st.metric("Mean Residual Change", f"{np.mean(residual_changes):.3f}")
            st.metric("Min Expected", f"{bus_data['expected'].min():.2f}")
            st.metric("Max Expected", f"{bus_data['expected'].max():.2f}")

    for i, bus_info in enumerate(bus_status_list):
        status_display, color = get_status_display(bus_info['Status'])
        if curr_status and status_display != curr_status:
            continue
        if license_plate and not (license_plate in bus_info["Bus"]):
            continue
        st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    padding: 0.6rem;
                    border-radius: 0.4rem;
                ">
                """,
                unsafe_allow_html=True,
        )
        
        if st.button(
            f"{bus_info['Bus']} | {status_display} | {bus_info['Trip Count']} trips | Max Change: {round(bus_info['Max Residual Change'], 2)}",
            use_container_width=True,
            key = bus_info["Bus"]
        ):
            modal(bus_info["Bus"])
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
