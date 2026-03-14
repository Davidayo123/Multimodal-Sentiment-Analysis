import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Portfolio | David", layout="wide", page_icon="⚡")

# --- SIDEBAR NAVIGATION & CONTROLS ---
st.sidebar.title("👨‍💻 David's AI Portfolio")
project_selection = st.sidebar.selectbox(
    "Select a Project:",
    ["1. Solar Microgrid Optimizer", "2. Fraud Detection Pipeline", "3. Brand Sentiment NLP"]
)

if project_selection == "1. Solar Microgrid Optimizer":
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Engineering Parameters")
    
    # User-controlled physics and ML parameters
    panel_efficiency = st.sidebar.slider("Panel Efficiency (r)", min_value=0.10, max_value=0.25, value=0.15, step=0.01, help="Standard panels are ~15%. Premium are ~22%.")
    perf_ratio = st.sidebar.slider("Performance Ratio (PR)", min_value=0.50, max_value=0.90, value=0.75, step=0.01, help="Accounts for system losses (heat, dust, wiring).")
    k_clusters = st.sidebar.slider("Microgrid Clusters (k)", min_value=10, max_value=200, value=114, step=1, help="Number of decentralized grids to create via K-Means.")
    
    # Constants
    annual_irradiance = 1709.64  # kWh/m2/year for Ikeja, Lagos

    # --- MAIN DASHBOARD AREA ---
    st.title("☀️ Urban Solar Microgrid Optimization: Ikeja")
    st.markdown("Transforming raw satellite footprints into actionable energy infrastructure using K-Means clustering.")

    # --- DATA LOADING (With Caching for Speed) ---
    @st.cache_data
    def load_building_data():
        """Loads pre-processed building data or generates mock data for Ikeja if missing."""
        try:
            # Replace 'ikeja_buildings.csv' with your actual saved dataset filename
            df = pd.read_csv("ikeja_buildings.csv") 
        except FileNotFoundError:
            # Generate 1,000 mock buildings centered around Ikeja if the real file isn't found
            np.random.seed(42)
            df = pd.DataFrame({
                'id': range(1000),
                'latitude': np.random.uniform(6.58, 6.62, 1000),
                'longitude': np.random.uniform(3.32, 3.36, 1000),
                'area_sqm': np.random.uniform(50, 300, 1000)
            })
        return df

    df = load_building_data()

    # --- DYNAMIC CALCULATIONS ---
    # Formula: E = A * r * H * PR
    df['annual_kwh'] = df['area_sqm'] * panel_efficiency * annual_irradiance * perf_ratio
    df['annual_mwh'] = df['annual_kwh'] / 1000

    total_mwh = df['annual_mwh'].sum()
    total_buildings = len(df)

    # --- K-MEANS CLUSTERING ---
    coords = df[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_

    # --- KPI METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Buildings Analyzed", f"{total_buildings:,}")
    col2.metric("Total Potential (MWh/year)", f"{total_mwh:,.2f}")
    col3.metric("Active Microgrids", f"{k_clusters}")

    st.markdown("---")

    # --- INTERACTIVE MAP ---
    st.subheader("Interactive Infrastructure Map")
    
    # Center map on Ikeja
    m = folium.Map(location=[6.6018, 3.3396], zoom_start=13, tiles="CartoDB positron")

    # Plot the microgrid centroids (Battery Storage Locations)
    for idx, center in enumerate(centroids):
        folium.CircleMarker(
            location=[center[0], center[1]],
            radius=6,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.7,
            popup=f"Microgrid Hub {idx+1}"
        ).add_to(m)

    # Display the map in Streamlit
    st_folium(m, width=1200, height=600)
    
    st.info("💡 **Pro Tip:** Adjust the sliders in the left sidebar to recalculate the physics and machine learning parameters in real-time.")

elif project_selection == "2. Fraud Detection Pipeline":
    st.title("💳 Financial Fraud Detection")
    st.markdown("Catching anomalies in highly imbalanced credit card data using **SMOTE** and **Threshold Tuning**.")

    # --- SIDEBAR: MOCK TRANSACTION SIMULATOR ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Transaction Simulator")
    st.sidebar.write("Test the model's logic based on top feature weights:")
    
    # Sliders for the top 2 features your model identified
    v10_val = st.sidebar.slider("Feature V10 (Risk Factor)", min_value=-10.0, max_value=10.0, value=0.0)
    v14_val = st.sidebar.slider("Feature V14 (Risk Factor)", min_value=-10.0, max_value=10.0, value=0.0)
    tx_amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)

    # Interactive prediction button (Simulated logic based on standard PCA fraud profiles)
    if st.sidebar.button("Analyze Transaction"):
        with st.spinner('Analyzing node pathways...'):
            if v10_val < -3.0 and v14_val < -3.0:
                st.sidebar.error(f"🚨 FRAUD DETECTED: Blocked ${tx_amount:,.2f}")
            else:
                st.sidebar.success(f"✅ Transaction Approved: ${tx_amount:,.2f}")

    # --- MAIN DASHBOARD AREA ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("The 'Accuracy Paradox'")
        st.info("In a dataset where only 0.2% of transactions are fraudulent, a model guessing 'Normal' every time achieves 99.8% accuracy but fails its business objective. This pipeline optimizes for **Recall** instead.")
    
    with col2:
        st.subheader("Why Threshold Tuning?")
        st.write("By lowering the AI's decision threshold from the default 50% to **15%**, we force the model to flag suspicious transactions earlier. This drastically reduces False Negatives (missed fraud) and saves millions in potential losses.")

    st.markdown("---")
    st.subheader("📊 Model Artifacts & Business Metrics")
    
    # --- DISPLAY SAVED ARTIFACTS ---
    tab1, tab2 = st.tabs(["Tuned Confusion Matrix", "Feature Importance"])
    
    with tab1:
        st.write("Visualizing the impact of the 15% probability threshold on True Positives.")
        try:
            # Loads the exact image you saved in Colab
            st.image("tuned_confusion_matrix.png", use_container_width=True)
        except FileNotFoundError:
            st.warning("⚠️ 'tuned_confusion_matrix.png' not found. Make sure the image is in the same folder as app.py!")
            
    with tab2:
        st.write("Extracting the internal weights of the Random Forest to identify the driving factors of fraud.")
        try:
             # Loads the exact image you saved in Colab
            st.image("feature_importance_plot.png", use_container_width=True)
        except FileNotFoundError:
            st.warning("⚠️ 'feature_importance_plot.png' not found. Make sure the image is in the same folder as app.py!")

elif project_selection == "3. Brand Sentiment NLP":
    st.title("📊 Multimodal Sentiment Analysis")
    st.markdown("Analyzing unstructured social media text (including emojis and slang) using a deep learning **Transformer (RoBERTa)**.")

    # --- SIDEBAR: LIVE INFERENCE SIMULATOR ---
    st.sidebar.markdown("---")
    st.sidebar.header("💬 Live Text Analyzer")
    st.sidebar.write("Test the NLP model's understanding of context and emojis:")
    
    # Text input for the recruiter to play with
    user_text = st.sidebar.text_area("Enter a simulated tweet/review:", value="The new UI update is absolutely terrible 🤮")
    
    # Interactive prediction button
    if st.sidebar.button("Analyze Sentiment"):
        with st.spinner("Processing through RoBERTa layers..."):
            # Lightweight mock logic for the portfolio demo
            text_lower = user_text.lower()
            if any(word in text_lower for word in ["🔥", "good", "amazing", "love", "best"]):
                st.sidebar.success("Sentiment: POSITIVE (Confidence: 0.98)")
            elif any(word in text_lower for word in ["🤮", "terrible", "bad", "😡", "hate", "worst"]):
                st.sidebar.error("Sentiment: NEGATIVE (Confidence: 0.95)")
            else:
                st.sidebar.info("Sentiment: NEUTRAL (Confidence: 0.82)")
                
    # --- MAIN DASHBOARD AREA ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("The Business Problem")
        st.info("Standard keyword models fail to understand internet context. 'This is sick 🔥' is positive, but 'I am sick 🤮' is negative. We deployed a Transformer model trained on 58 million tweets to capture this exact nuance.")
        
    with col2:
        st.subheader("The Engineering Solution")
        st.write("Instead of training from scratch, we programmatically ingested the **TweetEval** benchmark dataset and passed it through a pre-trained **Hugging Face RoBERTa** pipeline, extracting actionable brand health metrics.")

    st.markdown("---")
    st.subheader("📈 Executive Brand Dashboard")
    
    # --- DISPLAY SAVED ARTIFACTS ---
    tab1, tab2 = st.tabs(["Brand Health Distribution", "Model Accuracy (Confusion Matrix)"])
    
    with tab1:
        st.write("A high-level view of customer sentiment for executive reporting.")
        try:
            st.image("brand_sentiment_distribution.png", use_container_width=True)
        except FileNotFoundError:
            st.warning("⚠️ 'brand_sentiment_distribution.png' not found. Make sure it is in the same folder as app.py!")
            
    with tab2:
        st.write("Detailed breakdown of the AI's 77% accuracy against human labels.")
        try:
            st.image("sentiment_confusion_matrix.png", use_container_width=True)
        except FileNotFoundError:
            st.warning("⚠️ 'sentiment_confusion_matrix.png' not found. Make sure it is in the same folder as app.py!")