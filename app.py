import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL AND DATA ----------------
model = pickle.load(open("model.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("df_sql.csv")

df = load_data()

# ---------------- SESSION STATE FOR PAGE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"  # default page

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("RSP Navigation")

# Sidebar buttons as boxes
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"
if st.sidebar.button("📊 Data Analysis"):
    st.session_state.page = "Data Analysis"
if st.sidebar.button("🛒 Prediction"):
    st.session_state.page = "Prediction"

page_name = st.session_state.page  # current page

# ---------------- PAGE LOGIC ----------------
if page_name == "Home":
    st.title("Home Page")
    st.write("This is where your Home page code goes.")
elif page_name == "Data Analysis":
    st.title("Data Analysis Page")
    st.write("This is where your Data Analysis page code goes.")
elif page_name == "Prediction":
    st.title("Prediction Page")
    st.write("This is where your Prediction page code goes.")
# ---------------- HOME PAGE ----------------
if page_name == "Home":

    st.title("Retail Store Sales Forecasting System")

    st.write("""
    This project predicts **weekly retail store sales**
    using Machine Learning.

    Features used in prediction include:
    - Store
    - Department
    - Temperature
    - Fuel Price
    - CPI
    - Unemployment
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())


# ---------------- DATA ANALYSIS PAGE ----------------
# ---------------- DATA ANALYSIS PAGE ----------------
elif page_name == "Data Analysis":
    st.title("Sales Data Analysis")

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${df['Weekly_Sales'].sum():,.0f}")
    col2.metric("Average Weekly Sales", f"${df['Weekly_Sales'].mean():,.0f}")
    col3.metric("Total Stores", df["Store"].nunique())

    # ---------------- TOP 10 STORES ----------------
    st.subheader("Top 10 Stores by Total Sales")
    store_sales = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(8,5))
    store_sales.head(10).plot(kind="bar", ax=ax1, color='skyblue')
    ax1.set_ylabel("Total Sales")
    ax1.set_xlabel("Store")
    ax1.set_title("Top 10 Stores by Sales")
    st.pyplot(fig1)

    # ---------------- TOP 10 DEPARTMENTS ----------------
    st.subheader("Top 10 Departments by Total Sales")
    dept_sales = df.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    dept_sales.head(10).plot(kind="bar", ax=ax2, color='salmon')
    ax2.set_ylabel("Total Sales")
    ax2.set_xlabel("Department")
    ax2.set_title("Top 10 Departments by Sales")
    st.pyplot(fig2)

    # ---------------- TEMPERATURE VS SALES ----------------
    st.subheader("Temperature vs Weekly Sales")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.scatter(df["Temperature"], df["Weekly_Sales"], alpha=0.5)
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel("Weekly Sales")
    ax3.set_title("Temperature vs Weekly Sales")
    st.pyplot(fig3)

    # ---------------- WEEKLY SALES TREND ----------------
    st.subheader("Weekly Sales Trend Over Time")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    sales_trend = df.groupby("Date")["Weekly_Sales"].sum()
    fig4, ax4 = plt.subplots(figsize=(10,5))
    sales_trend.plot(ax=ax4)
    ax4.set_ylabel("Total Sales")
    ax4.set_title("Sales Trend Over Time")
    st.pyplot(fig4)

    # ---------------- CORRELATION HEATMAP ----------------
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig5, ax5 = plt.subplots(figsize=(10,6))
    import seaborn as sns
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax5)
    ax5.set_title("Correlation Between Numerical Features")
    st.pyplot(fig5)



# ---------------- PREDICTION PAGE ----------------
# ---------------- PREDICTION PAGE ----------------
# ---------------- PREDICTION PAGE ----------------
# ---------------- PREDICTION PAGE ----------------
# ---------------- PREDICTION PAGE ----------------
elif page_name == "Prediction":
    # ---------------- PAGE HEADER ----------------
    st.markdown(
        """
        <h1 style='background: linear-gradient(to right, #ff7e5f, #feb47b); 
        -webkit-background-clip: text; color: transparent; text-align:center'>
        🛒 Retail Sales Prediction
        </h1>
        <p style='text-align:center; font-size:16px; color:#555;'>Predict weekly sales based on store, department, promotions, and economic indicators.</p>
        """,
        unsafe_allow_html=True
    )

    # ---------------- STORE & DEPARTMENT INFO ----------------
    st.markdown(
        """
        <div style='background-color:#f0f8ff;padding:15px;border-radius:10px;margin-bottom:10px'>
        <h3 style='color:#003366;'>🏬 Store & Department Info</h3>
        </div>
        """, unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    store = col1.selectbox("Store", sorted(df["Store"].unique()))
    dept = col2.selectbox("Department", sorted(df["Dept"].unique()))
    type_store_str = col3.selectbox("Store Type", ["A", "B", "C"])
    size = col3.number_input("Store Size (sq.ft)", value=150000, step=1000)

    # Map Store Type to numeric
    type_map = {"A": 1, "B": 2, "C": 3}
    type_store = type_map[type_store_str]

    # ---------------- ECONOMIC & ENVIRONMENTAL INDICATORS ----------------
    st.markdown(
        """
        <div style='background-color:#e8f5e9;padding:15px;border-radius:10px;margin-bottom:10px'>
        <h3 style='color:#2e7d32;'>📊 Economic & Environmental Indicators</h3>
        </div>
        """, unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    temperature = col1.slider("Temperature (°F)", 0.0, 120.0, 70.0, 0.5, help="Temperature at the store location")
    fuel_price = col2.slider("Fuel Price ($)", 0.0, 10.0, 3.0, 0.01, help="Average fuel price in the region")
    cpi = col3.slider("CPI", 50.0, 500.0, 200.0, 0.1, help="Consumer Price Index for the week")
    unemployment = st.slider("Unemployment Rate (%)", 0.0, 20.0, 8.0, 0.1, help="Unemployment rate in percent")

    # ---------------- PROMOTIONAL MARKDOWNS ----------------
    st.markdown(
        """
        <div style='background-color:#fff3e0;padding:15px;border-radius:10px;margin-bottom:10px'>
        <h3 style='color:#e65100;'>💸 Weekly Promotional Markdown Amounts</h3>
        </div>
        """, unsafe_allow_html=True
    )
    markdown_values = []
    cols = st.columns(5)
    for i, col in enumerate(cols, start=1):
        value = col.number_input(f"MarkDown{i}", 0.0, 50000.0, 0.0, 0.1, help=f"Promotional discount for campaign {i}")
        markdown_values.append(value)
    markdown1, markdown2, markdown3, markdown4, markdown5 = markdown_values

    # ---------------- TIME & HOLIDAY ----------------
    st.markdown(
        """
        <div style='background-color:#f3e5f5;padding:15px;border-radius:10px;margin-bottom:10px'>
        <h3 style='color:#6a1b9a;'>🗓️ Time & Holiday Info</h3>
        </div>
        """, unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    isHoliday = col1.selectbox("Is Holiday?", [0, 1], help="Whether this week contains a store holiday")
    year = col2.number_input("Year", 2010, 2030, 2012)
    month = col2.number_input("Month", 1, 12, 1)
    week = col3.number_input("Week of Year", 1, 53, 1)

    # ---------------- PREDICT BUTTON ----------------
    if st.button("Predict Weekly Sales"):
        # Feature alignment
        feature_names = model.feature_names_in_
        input_data = pd.DataFrame([{
            "Store": store,
            "Dept": dept,
            "IsHoliday_x": isHoliday,
            "IsHoliday_y": isHoliday,
            "Temperature": temperature,
            "Fuel_Price": fuel_price,
            "MarkDown1": markdown1,
            "MarkDown2": markdown2,
            "MarkDown3": markdown3,
            "MarkDown4": markdown4,
            "MarkDown5": markdown5,
            "CPI": cpi,
            "Unemployment": unemployment,
            "Type": type_store,
            "Size": size,
            "Year": year,
            "Month": month,
            "Week": week
        }])
        input_data = input_data[feature_names]

        # Make prediction
        prediction = model.predict(input_data)[0]

        # ---------------- DISPLAY PREDICTION ----------------
        st.markdown(
            """
            <div style='background-color:#e1f5fe;padding:15px;border-radius:10px;margin-top:10px'>
            <h3 style='color:#0277bd;'>💰 Predicted Weekly Sales</h3>
            </div>
            """, unsafe_allow_html=True
        )

        # Color-coded output
        if prediction > 20000:
            st.markdown(f"<h2 style='color:green'>${prediction:,.2f} ✅</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:orange'>${prediction:,.2f} ⚠️</h2>", unsafe_allow_html=True)

        # Custom progress bar visualization
        progress_value = min(prediction / 50000, 1.0) * 100
        st.markdown(f"""
        <div style='background-color:#e0e0e0;width:100%;border-radius:10px'>
            <div style='width:{progress_value}%;background-color:#29b6f6;padding:10px;border-radius:10px;color:white;text-align:center'>
                {prediction:,.0f} USD
            </div>
        </div>
        """, unsafe_allow_html=True)