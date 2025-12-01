import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

# Page configuration
st.set_page_config(page_title="Indian Crime Analytics", layout="wide", page_icon="🚔")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚔 Indian Crime Data Analytics & ML Platform")
st.markdown("### Comprehensive Analysis of Crime Statistics Across Indian States")

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Select Analysis", 
                        ["Data Overview", "Exploratory Analysis", "Crime Prediction", 
                         "State Clustering", "Crime Trends", "Custom Analysis"])

# Load data
@st.cache_data
def load_data():
    # Sample data - will be replaced by uploaded file
    data = {
        'STATE/UT': ['SAMPLE STATE'],
        'DISTRICT': ['SAMPLE DISTRICT'],
        'YEAR': [2001],
        'MURDER': [0]
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    # Clean column names - remove extra spaces
    df.columns = df.columns.str.strip()
    
    # Create TOTAL_IPC if it doesn't exist by summing all numeric crime columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['YEAR']
    crime_numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if 'TOTAL IPC' not in df.columns and 'TOTAL_IPC' not in df.columns:
        df['TOTAL_IPC'] = df[crime_numeric_cols].sum(axis=1)
        st.sidebar.info("Created TOTAL_IPC column by summing all crime types")
    elif 'TOTAL IPC' in df.columns:
        df['TOTAL_IPC'] = df['TOTAL IPC']
    
else:
    st.warning("⚠️ Please upload a CSV file to begin analysis")
    st.info("Upload your crime data CSV file using the sidebar to get started!")
    st.stop()

# Data preprocessing
crime_columns = [col for col in df.columns if col not in ['STATE/UT', 'DISTRICT', 'YEAR']]

# Ensure TOTAL_IPC exists
if 'TOTAL_IPC' not in df.columns:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    crime_numeric_cols = [col for col in numeric_cols if col != 'YEAR']
    df['TOTAL_IPC'] = df[crime_numeric_cols].sum(axis=1)
    crime_columns.append('TOTAL_IPC')

# PAGE 1: DATA OVERVIEW
if page == "Data Overview":
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("States/UTs", df['STATE/UT'].nunique())
    with col3:
        st.metric("Districts", df['DISTRICT'].nunique())
    with col4:
        st.metric("Years Covered", df['YEAR'].nunique())
    
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(df[crime_columns].describe(), use_container_width=True)
    
    st.subheader("Missing Values Analysis")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        fig = px.bar(x=missing.index, y=missing.values, labels={'x': 'Columns', 'y': 'Missing Count'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values detected!")

# PAGE 2: EXPLORATORY ANALYSIS
elif page == "Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Crime Distribution", "Geographic Analysis", "Correlation Matrix"])
    
    with tab1:
        st.subheader("Top Crime Types")
        crime_totals = df[crime_columns].sum().sort_values(ascending=False).head(15)
        fig = px.bar(x=crime_totals.index, y=crime_totals.values, 
                     labels={'x': 'Crime Type', 'y': 'Total Cases'},
                     title="Top 15 Crime Categories",
                     color=crime_totals.values,
                     color_continuous_scale='Reds')
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Crime Distribution by Year")
        yearly = df.groupby('YEAR')['TOTAL_IPC'].sum().reset_index()
        fig2 = px.line(yearly, x='YEAR', y='TOTAL_IPC', markers=True,
                       title="Total Crimes Over Years")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("State-wise Crime Analysis")
        state_crimes = df.groupby('STATE/UT')['TOTAL_IPC'].sum().sort_values(ascending=False).head(20)
        fig = px.bar(x=state_crimes.index, y=state_crimes.values,
                     labels={'x': 'State/UT', 'y': 'Total Crimes'},
                     title="Top 20 States by Total Crimes",
                     color=state_crimes.values,
                     color_continuous_scale='Viridis')
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("District-wise Analysis")
        selected_state = st.selectbox("Select State", df['STATE/UT'].unique())
        district_data = df[df['STATE/UT'] == selected_state].groupby('DISTRICT')['TOTAL_IPC'].sum().sort_values(ascending=False)
        fig3 = px.bar(x=district_data.index, y=district_data.values,
                      title=f"Crime Distribution in {selected_state}")
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.subheader("Crime Correlation Matrix")
        selected_crimes = st.multiselect("Select crimes to analyze", 
                                        crime_columns[:15], 
                                        default=crime_columns[:8])
        if selected_crimes:
            corr_matrix = df[selected_crimes].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           color_continuous_scale='RdBu_r',
                           title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

# PAGE 3: CRIME PREDICTION
elif page == "Crime Prediction":
    st.header("🎯 Crime Prediction Model")
    
    st.info("This model predicts total crimes (TOTAL_IPC) based on individual crime categories and historical patterns.")
    
    # Feature selection
    feature_cols = [col for col in crime_columns if col != 'TOTAL_IPC' and col not in ['STATE/UT', 'DISTRICT', 'YEAR']]
    
    if len(feature_cols) == 0:
        st.error("No feature columns available for prediction. Please upload a valid dataset.")
        st.stop()
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['TOTAL_IPC'].fillna(0)
    
    # Train-test split
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])
    with col2:
        n_estimators = st.slider("Number of Estimators", 50, 300, 100)
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            else:
                model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            st.success("Model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Prediction vs Actual
            st.subheader("Prediction vs Actual Values")
            comparison_df = pd.DataFrame({
                'Actual': y_test[:50],
                'Predicted': y_pred[:50]
            }).reset_index(drop=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=comparison_df['Actual'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(y=comparison_df['Predicted'], mode='lines+markers', name='Predicted'))
            fig.update_layout(title="First 50 Predictions", xaxis_title="Sample", yaxis_title="Total Crimes")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig2 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                             title="Top 15 Important Features")
                st.plotly_chart(fig2, use_container_width=True)

# PAGE 4: STATE CLUSTERING
elif page == "State Clustering":
    st.header("🗺️ State Clustering Analysis")
    st.write("Group states with similar crime patterns using K-Means clustering")
    
    # Aggregate by state
    state_data = df.groupby('STATE/UT')[crime_columns].sum().reset_index()
    
    n_clusters = st.slider("Number of Clusters", 2, 10, 4)
    
    if st.button("Perform Clustering", type="primary"):
        with st.spinner("Clustering states..."):
            # Prepare data
            X = state_data[crime_columns].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            state_data['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Display results
            st.subheader("Clustering Results")
            for i in range(n_clusters):
                cluster_states = state_data[state_data['Cluster'] == i]['STATE/UT'].tolist()
                st.write(f"**Cluster {i+1}** ({len(cluster_states)} states): {', '.join(cluster_states)}")
            
            # Visualization
            st.subheader("Cluster Visualization")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plot_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'State': state_data['STATE/UT'],
                'Cluster': state_data['Cluster'].astype(str)
            })
            
            fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                           hover_data=['State'], title="State Clusters (PCA Visualization)")
            st.plotly_chart(fig, use_container_width=True)

# PAGE 5: CRIME TRENDS
elif page == "Crime Trends":
    st.header("📈 Crime Trends Analysis")
    
    selected_crime = st.selectbox("Select Crime Type", crime_columns)
    
    # Yearly trends
    st.subheader(f"Yearly Trend: {selected_crime}")
    yearly_trend = df.groupby('YEAR')[selected_crime].sum().reset_index()
    fig1 = px.line(yearly_trend, x='YEAR', y=selected_crime, markers=True,
                   title=f"{selected_crime} Over Years")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Top states
    st.subheader(f"Top 10 States by {selected_crime}")
    state_trend = df.groupby('STATE/UT')[selected_crime].sum().sort_values(ascending=False).head(10)
    fig2 = px.bar(x=state_trend.index, y=state_trend.values,
                  labels={'x': 'State', 'y': selected_crime},
                  color=state_trend.values,
                  color_continuous_scale='Reds')
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Year-over-year comparison
    st.subheader("Year-over-Year Comparison")
    state_selected = st.selectbox("Select State for Comparison", df['STATE/UT'].unique())
    state_yearly = df[df['STATE/UT'] == state_selected].groupby('YEAR')[crime_columns[:8]].sum()
    
    fig3 = go.Figure()
    for col in state_yearly.columns:
        fig3.add_trace(go.Scatter(x=state_yearly.index, y=state_yearly[col],
                                  mode='lines+markers', name=col))
    fig3.update_layout(title=f"Crime Trends in {state_selected}", xaxis_title="Year", yaxis_title="Cases")
    st.plotly_chart(fig3, use_container_width=True)

# PAGE 6: CUSTOM ANALYSIS
elif page == "Custom Analysis":
    st.header("🔧 Custom Analysis")
    
    analysis_type = st.radio("Select Analysis Type", 
                             ["Compare Multiple States", "Compare Crime Types", "Custom Filter"])
    
    if analysis_type == "Compare Multiple States":
        states = st.multiselect("Select States to Compare", df['STATE/UT'].unique(), 
                               default=list(df['STATE/UT'].unique())[:3])
        crime_type = st.selectbox("Crime Type", crime_columns)
        
        if states:
            compare_data = df[df['STATE/UT'].isin(states)].groupby(['STATE/UT', 'YEAR'])[crime_type].sum().reset_index()
            fig = px.line(compare_data, x='YEAR', y=crime_type, color='STATE/UT',
                         title=f"{crime_type} Comparison", markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Compare Crime Types":
        state = st.selectbox("Select State", df['STATE/UT'].unique())
        crimes = st.multiselect("Select Crimes", crime_columns, default=crime_columns[:5])
        
        if crimes:
            crime_data = df[df['STATE/UT'] == state].groupby('YEAR')[crimes].sum()
            fig = go.Figure()
            for crime in crimes:
                fig.add_trace(go.Bar(x=crime_data.index, y=crime_data[crime], name=crime))
            fig.update_layout(title=f"Crime Comparison in {state}", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.subheader("Custom Data Filter")
        col1, col2 = st.columns(2)
        with col1:
            year_filter = st.multiselect("Filter by Year", df['YEAR'].unique(), default=df['YEAR'].unique())
        with col2:
            state_filter = st.multiselect("Filter by State", df['STATE/UT'].unique())
        
        filtered_df = df[df['YEAR'].isin(year_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df['STATE/UT'].isin(state_filter)]
        
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button("Download Filtered Data", 
                          filtered_df.to_csv(index=False), 
                          "filtered_data.csv", 
                          "text/csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Upload your own CSV file to analyze custom crime data!")
st.sidebar.markdown("Built with ❤️ using Streamlit")