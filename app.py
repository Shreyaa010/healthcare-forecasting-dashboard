import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Healthcare Forecasting Dashboard",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.dashboard-title {
    text-align:center;
    color:#1565c0;
    font-size:40px;
    font-weight:bold;
}

.dashboard-subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.kpi-card {
    background-color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown("""
<div class="dashboard-title">
🏥 Healthcare Forecasting Dashboard
</div>

<div class="dashboard-subtitle">
Data Analytics & Machine Learning Project
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

cols = [
    'Children apprehended and placed in CBP custody*',
    'Children in CBP custody',
    'Children transferred out of CBP custody',
    'Children in HHS Care',
    'Children discharged from HHS Care'
]

for col in cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=True)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📅 Filter Data")

start_date = st.sidebar.date_input(
    "Start Date",
    df['Date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    df['Date'].max()
)

filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date))
    &
    (df['Date'] <= pd.to_datetime(end_date))
].copy()

if len(filtered_df) < 2:
    st.error("Please select a larger date range.")
    st.stop()

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

latest = int(filtered_df['Children in HHS Care'].iloc[-1])
average = int(filtered_df['Children in HHS Care'].mean())
maximum = int(filtered_df['Children in HHS Care'].max())

st.subheader("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
        <h3>Latest Care Load</h3>
        <h2>{latest:,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
        <h3>Average Care Load</h3>
        <h2>{average:,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card">
        <h3>Maximum Care Load</h3>
        <h2>{maximum:,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# --------------------------------------------------
# MACHINE LEARNING PREDICTION
# --------------------------------------------------

filtered_df = filtered_df.sort_values('Date')

filtered_df['day_num'] = np.arange(len(filtered_df))

X = filtered_df[['day_num']]
y = filtered_df['Children in HHS Care']

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(
    len(filtered_df),
    len(filtered_df) + 7
).reshape(-1, 1)

future_predictions = model.predict(future_days)

last_date = filtered_df['Date'].iloc[-1]

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7
)

prediction_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Care Load": future_predictions.astype(int)
})

# --------------------------------------------------
# FORECAST GRAPH
# --------------------------------------------------

st.subheader("📈 Healthcare Care Load Forecast")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    filtered_df['Date'],
    filtered_df['Children in HHS Care'],
    marker='o',
    linewidth=2,
    label='Actual'
)

ax.plot(
    prediction_df['Date'],
    prediction_df['Predicted Care Load'],
    marker='o',
    linestyle='--',
    linewidth=2,
    label='Predicted'
)

ax.set_title("Healthcare Care Load Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Children Count")
ax.legend()

plt.xticks(rotation=45)

st.pyplot(fig)

st.markdown("---")

# --------------------------------------------------
# MONTHLY CHART
# --------------------------------------------------

st.subheader("📊 Monthly Care Load Distribution")

monthly_data = (
    filtered_df
    .groupby(filtered_df['Date'].dt.month)
    ['Children in HHS Care']
    .mean()
)

st.bar_chart(monthly_data)

st.markdown("---")

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------

st.subheader("📋 Filtered Dataset")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)

# --------------------------------------------------
# PREDICTION TABLE
# --------------------------------------------------

st.subheader("🔮 Future Prediction (Next 7 Days)")

st.dataframe(
    prediction_df,
    use_container_width=True
)

# --------------------------------------------------
# DOWNLOAD BUTTON
# --------------------------------------------------

st.subheader("📥 Export Data")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="healthcare_data.csv",
    mime="text/csv"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Developed by Shreya Sorathiya | Healthcare Forecasting Dashboard"
)