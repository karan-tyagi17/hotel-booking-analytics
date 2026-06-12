import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Hotel Booking Analytics",
    layout="wide"
)

df = pd.read_csv('data/processed/hotel_bookings_cleaned.csv')

st.title(" Hotel Booking Analytics Dashboard")
st.write("Analyzing 119,390 hotel bookings to understand demand, cancellations and revenue patterns.")
# ─── SIDEBAR FILTERS ────────
st.sidebar.header("Filters")

# Hotel filter
hotel_options = ["All"] + list(df['hotel'].unique())
selected_hotel = st.sidebar.selectbox("Select Hotel", hotel_options)

# Month filter
month_options = ["All"] + list(df['arrival_date_month'].unique())
selected_month = st.sidebar.selectbox("Select Month", month_options)

# Apply filters to data
filtered_df = df.copy()

if selected_hotel != "All":
    filtered_df = filtered_df[filtered_df['hotel'] == selected_hotel]

if selected_month != "All":
    filtered_df = filtered_df[filtered_df['arrival_date_month'] == selected_month]

# ─── KPI CARDS ────────────────────
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Bookings", f"{len(filtered_df):,}")

with col2:
    cancel_rate = round(filtered_df['is_canceled'].mean() * 100, 2)
    st.metric("Cancellation Rate", f"{cancel_rate}%")

with col3:
    avg_adr = round(filtered_df['adr'].mean(), 2)
    st.metric("Avg Daily Rate", f"€{avg_adr}")

with col4:
    avg_lead = round(filtered_df['lead_time'].mean(), 0)
    st.metric("Avg Lead Time", f"{int(avg_lead)} days")

# ─── CHARTS ──────────────────
st.subheader("Booking Trends")

# Monthly bookings chart
monthly = filtered_df.groupby(
    ['arrival_month_num', 'arrival_date_month', 'hotel']
).size().reset_index(name='total_bookings')

monthly = monthly.sort_values('arrival_month_num')

fig = px.line(
    monthly,
    x='arrival_date_month',
    y='total_bookings',
    color='hotel',
    title='Monthly Bookings by Hotel Type',
    labels={
        'arrival_date_month': 'Month',
        'total_bookings': 'Total Bookings'
    },
    markers=True
)

st.plotly_chart(fig, use_container_width=True)
    
# ─── CANCELLATION ANALYSIS ─────────────────────────
st.subheader("Cancellation Analysis")

col1, col2 = st.columns(2)

with col1:
    # Cancellation by market segment
    segment = filtered_df.groupby('market_segment')['is_canceled'].mean().reset_index()
    segment['cancellation_rate'] = round(segment['is_canceled'] * 100, 2)
    segment = segment.sort_values('cancellation_rate', ascending=False)

    fig2 = px.bar(
        segment,
        x='market_segment',
        y='cancellation_rate',
        title='Cancellation Rate by Market Segment (%)',
        labels={'market_segment': 'Market Segment', 'cancellation_rate': 'Cancellation Rate (%)'},
        text='cancellation_rate',
        color='cancellation_rate',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # Cancellation by deposit type
    deposit = filtered_df.groupby('deposit_type')['is_canceled'].mean().reset_index()
    deposit['cancellation_rate'] = round(deposit['is_canceled'] * 100, 2)
    deposit = deposit.sort_values('cancellation_rate', ascending=False)

    fig3 = px.bar(
        deposit,
        x='deposit_type',
        y='cancellation_rate',
        title='Cancellation Rate by Deposit Type (%)',
        labels={'deposit_type': 'Deposit Type', 'cancellation_rate': 'Cancellation Rate (%)'},
        text='cancellation_rate',
        color='cancellation_rate',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig3, use_container_width=True)

# Lead time and special requests
col3, col4 = st.columns(2)

with col3:
    lead = filtered_df.copy()
    lead['lead_time_band'] = pd.cut(
        lead['lead_time'],
        bins=[0, 7, 30, 90, 180, 999],
        labels=['0-7 days', '8-30 days', '31-90 days', '91-180 days', '180+ days']
    )
    lead_cancel = lead.groupby('lead_time_band', observed=True)['is_canceled'].mean().reset_index()
    lead_cancel['cancellation_rate'] = round(lead_cancel['is_canceled'] * 100, 2)

    fig4 = px.bar(
        lead_cancel,
        x='lead_time_band',
        y='cancellation_rate',
        title='Cancellation Rate by Lead Time (%)',
        labels={'lead_time_band': 'Lead Time', 'cancellation_rate': 'Cancellation Rate (%)'},
        text='cancellation_rate',
        color='cancellation_rate',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    special = filtered_df.groupby('total_of_special_requests')['is_canceled'].mean().reset_index()
    special['cancellation_rate'] = round(special['is_canceled'] * 100, 2)

    fig5 = px.bar(
        special,
        x='total_of_special_requests',
        y='cancellation_rate',
        title='Cancellation Rate by Special Requests (%)',
        labels={'total_of_special_requests': 'Special Requests', 'cancellation_rate': 'Cancellation Rate (%)'},
        text='cancellation_rate',
        color='cancellation_rate',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig5, use_container_width=True)

# ─── REVENUE ANALYSIS ──────────────────────────────
st.subheader("Revenue Analysis")

col5, col6 = st.columns(2)

with col5:
    adr_monthly = filtered_df[filtered_df['is_canceled'] == 0].groupby(
        ['arrival_month_num', 'arrival_date_month', 'hotel']
    )['adr'].mean().reset_index()
    adr_monthly = adr_monthly.sort_values('arrival_month_num')
    adr_monthly['adr'] = round(adr_monthly['adr'], 2)

    fig6 = px.line(
        adr_monthly,
        x='arrival_date_month',
        y='adr',
        color='hotel',
        title='Average Daily Rate by Hotel and Month (€)',
        labels={'arrival_date_month': 'Month', 'adr': 'Avg Daily Rate (€)'},
        markers=True
    )
    st.plotly_chart(fig6, use_container_width=True)

with col6:
    value_seg = filtered_df.groupby('market_segment').agg(
        avg_daily_rate=('adr', 'mean'),
        cancellation_rate=('is_canceled', 'mean')
    ).reset_index()
    value_seg['true_value'] = round(
        value_seg['avg_daily_rate'] * (1 - value_seg['cancellation_rate']), 2
    )
    value_seg = value_seg.sort_values('true_value', ascending=False)

    fig7 = px.bar(
        value_seg,
        x='market_segment',
        y='true_value',
        title='True Value per Booking by Segment (€)',
        labels={'market_segment': 'Market Segment', 'true_value': 'True Value (€)'},
        text='true_value',
        color='true_value',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig7, use_container_width=True)

# ─── COUNTRY ANALYSIS ──────────────────────────────
st.subheader("Top 10 Countries")

countries = filtered_df[filtered_df['country'] != 'Unknown'].groupby('country').agg(
    total_bookings=('is_canceled', 'count'),
    cancellation_rate=('is_canceled', 'mean')
).reset_index()
countries['cancellation_rate'] = round(countries['cancellation_rate'] * 100, 2)
countries = countries.sort_values('total_bookings', ascending=False).head(10)

fig8 = px.bar(
    countries,
    x='country',
    y='total_bookings',
    color='cancellation_rate',
    title='Top 10 Countries by Bookings (colored by cancellation rate)',
    labels={'country': 'Country', 'total_bookings': 'Total Bookings'},
    text='total_bookings',
    color_continuous_scale='Reds'
)
st.plotly_chart(fig8, use_container_width=True)