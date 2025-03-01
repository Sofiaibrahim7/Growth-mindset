import streamlit as st
import pandas as pd
import os
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import plotly.express as px
import pyarrow as pa
import pyarrow.parquet as pq

st.set_page_config(page_title="Data Sweeper", layout='wide')

# Sidebar: Theme Selection
mode = st.sidebar.radio("Select Theme", ["Dark Mode", "Light Mode"])
if mode == "Dark Mode":
    st.markdown("""
        <style>
            .stApp { background-color: black; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            .stApp { background-color: white; color: black; }
        </style>
    """, unsafe_allow_html=True)

st.title("🧹 Data Sweeper - Advanced Data Cleaning & Visualization")
st.write("🚀 Upload CSV/Excel files, clean, visualize, and export in different formats.")

# Upload Files
uploaded_files = st.file_uploader("📂 Upload Files (CSV/Excel):", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    merged_df = pd.DataFrame()
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[-1].lower()
        df = pd.read_csv(file) if file_ext == ".csv" else pd.read_excel(file)
        df.columns = df.columns.str.strip()
        
        # Handle Missing Headers
        if all(df.columns.str.contains("Unnamed")):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        
        # Merge Files
        merged_df = pd.concat([merged_df, df], ignore_index=True) if not merged_df.empty else df
        
    # Display Data Preview
    st.subheader("👀 Merged Data Preview")
    st.dataframe(merged_df.head())
    
    # Data Cleaning Options
    if st.button("🗑️ Remove Duplicates"):
        merged_df.drop_duplicates(inplace=True)
        st.success("✅ Duplicates Removed!")
    
    if st.button("🔧 Fill Missing Values"):
        numeric_cols = merged_df.select_dtypes(include=['number']).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].mean())
        st.success("✅ Missing Values Filled!")
    
    # Column Selection
    selected_columns = st.multiselect("📌 Select Columns to Display", merged_df.columns, default=merged_df.columns)
    merged_df = merged_df[selected_columns]
    
    # Data Scaling
    scaling_method = st.selectbox("Choose Scaling Method", ["None", "Normalization (Min-Max)", "Standardization (Z-Score)"])
    numeric_cols = merged_df.select_dtypes(include=['number']).columns
    if scaling_method == "Normalization (Min-Max)":
        scaler = MinMaxScaler()
        merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])
    elif scaling_method == "Standardization (Z-Score)":
        scaler = StandardScaler()
        merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])
    
    # Outlier Detection
    if st.checkbox("🔍 Detect Outliers"):
        Q1 = merged_df[numeric_cols].quantile(0.25)
        Q3 = merged_df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers = merged_df[(merged_df[numeric_cols] < (Q1 - 1.5 * IQR)) | (merged_df[numeric_cols] > (Q3 + 1.5 * IQR))]
        st.write("⚠️ Detected Outliers:", outliers)
        st.dataframe(outliers)
    
    # Data Visualization
    chart_type = st.selectbox("📊 Choose Chart Type", ["None", "Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart"])
    if chart_type != "None":
        x_axis = st.selectbox("Select X-axis", merged_df.columns)
        y_axis = st.selectbox("Select Y-axis", merged_df.select_dtypes(include=['number']).columns)
        
        if chart_type == "Line Chart":
            st.line_chart(merged_df.set_index(x_axis)[y_axis])
        elif chart_type == "Bar Chart":
            fig = px.bar(merged_df, x=x_axis, y=y_axis, title="Bar Chart")
            st.plotly_chart(fig)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(merged_df, x=x_axis, y=y_axis, title="Scatter Plot")
            st.plotly_chart(fig)
        elif chart_type == "Pie Chart":
            fig = px.pie(merged_df, names=x_axis, values=y_axis, title="Pie Chart")
            st.plotly_chart(fig)
    
    # File Conversion
    conversion_type = st.radio("📂 Convert File to:", ["CSV", "Excel", "JSON", "Parquet"])
    if st.button("💾 Convert & Download"):
        buffer = BytesIO()
        file_name = "processed_data"
        
        if conversion_type == "CSV":
            merged_df.to_csv(buffer, index=False)
            file_name += ".csv"
            mime_type = "text/csv"
        elif conversion_type == "Excel":
            merged_df.to_excel(buffer, index=False)
            file_name += ".xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif conversion_type == "JSON":
            json_data = merged_df.to_json(orient='records', lines=True)
            buffer = BytesIO(json_data.encode())
            file_name += ".json"
            mime_type = "application/json"
        elif conversion_type == "Parquet":
            parquet_path = "data.parquet"
            pq.write_table(pa.Table.from_pandas(merged_df), parquet_path)
            with open(parquet_path, "rb") as f:
                st.download_button("⬇️ Download as Parquet", f, "data.parquet", "application/octet-stream")
            
        buffer.seek(0)
        st.download_button("⬇️ Download File", buffer, file_name, mime=mime_type)
    
    st.success("✅ Processing Complete!")
