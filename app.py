import streamlit as st
import pandas as pd
import joblib
from groq import Groq

st.set_page_config(page_title="Finance Analyzer", layout="wide")

#UI
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #000000);
    color: white;
}

/* Titles */
h1, h2, h3 {
    color: #e2e8f0;
}

/* Inputs */
.stNumberInput input {
    background-color: #B2BBD9;
    color: white;
    border-radius: 8px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #C93A5E);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #960227);
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

#Header
st.markdown("""
<h1 style='text-align: center; 
background: linear-gradient(90deg, #C93A5E); 
-webkit-background-clip: text; 
color: transparent;'>
AI Personal Finance Analyzer
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# API ID
client = Groq(api_key="Your API key")

#Loads the model
model = joblib.load("finance_model.pkl")

# UI for input
st.subheader("📥 Enter Your Financial Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("💵 Income", min_value=0.0, step=100.0)
    savings = st.number_input("💰 Savings", min_value=0.0, step=100.0)

with col2:
    food = st.number_input("🍔 Food", min_value=0.0, step=100.0)
    travel = st.number_input("🚗 Travel", min_value=0.0, step=100.0)
    shopping = st.number_input("🛍 Shopping", min_value=0.0, step=100.0)
    bills = st.number_input("📄 Bills", min_value=0.0, step=100.0)
    entertainment = st.number_input("🎬 Entertainment", min_value=0.0, step=100.0)

st.markdown("---")

if st.button("🚀 Analyze My Finances", use_container_width=True):

    # Validation
    if income <= 0:
        st.error("❌ Please enter valid income.")
        st.stop()

    if savings > income:
        st.error("❌ Savings cannot exceed income.")
        st.stop()

    total_expense = food + travel + shopping + bills + entertainment

    if total_expense == 0:
        st.warning("⚠ Please enter at least one expense.")
        st.stop()

    if total_expense > income:
        st.warning("⚠ You are spending more than your income!")

    # input data table
    m1, m2, m3 = st.columns(3)
    m1.metric("💵 Income", f"₹{income}")
    m2.metric("💸 Expenses", f"₹{total_expense}")
    m3.metric("💰 Savings", f"₹{savings}")

    # input to the ML model
    input_data = pd.DataFrame(
        [[income, food, travel, shopping, bills, entertainment, savings]],
        columns=["Income", "Food", "Travel", "Shopping", "Bills", "Entertainment", "Savings"]
    )

    prediction = model.predict(input_data)[0]

    st.markdown(f"### 📊 Risk Level: **{prediction}**")

    # Prompt
    prompt = f"""
    Income: {income}
    Expenses: {total_expense}
    Savings: {savings}
    Risk Level: {prediction}
    this are monthly expenses in rupees.

    Provide:
    🔍 Financial Analysis
    💡 3 Personalized Tips
    📊 Budget Strategy
    ⚠ Risk Warning
    🛡 Precautions

    Use numbers and avoid generic advice.
    """

    # GEN API call
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional financial advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        advice = response.choices[0].message.content

    except Exception as e:
        st.error(f"⚠ AI Error: {e}")
        advice = "⚠ Unable to generate AI advice. Please try again."

    #output
    st.markdown("### 🤖 AI Financial Report")
    st.markdown(
        f"""
        <div class="card">
        {advice}
        </div>
        """,
        unsafe_allow_html=True
    )

    #insigts
    if savings < 0.2 * income:
        st.warning("⚠ Savings are below 20% of income.")

    #graph
    st.markdown("### 📊 Expense Breakdown")

    expense_data = {
        "Food": food,
        "Travel": travel,
        "Shopping": shopping,
        "Bills": bills,
        "Entertainment": entertainment
    }

    chart = pd.DataFrame(expense_data.items(), columns=["Category", "Amount"])

    chart["Amount"] = pd.to_numeric(chart["Amount"], errors="coerce").fillna(0)

    if chart["Amount"].sum() == 0:
        st.warning("⚠ No valid expense data to display.")
    else:
        st.bar_chart(chart.set_index("Category"))
        # st.area_chart(chart.set_index("Category"))


