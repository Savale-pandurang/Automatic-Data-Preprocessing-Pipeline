import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Streamlit Page Config ---
st.set_page_config(page_title="Smart CSV Analyzer", page_icon="📊", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("🔎 Navigation")
menu = st.sidebar.radio("Go to:", ["📂 Upload Data", "🛠️ Preprocessing", "📉 Outlier Handling", "🤖 Modeling"])

# --- Global variables ---
if "df" not in st.session_state:
    st.session_state.df = None

# --- Upload Data ---
if menu == "📂 Upload Data":
    st.title("📂 CSV File Uploader & Preview")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # Save in session state
        st.success("✅ File uploaded successfully!")

        st.subheader("👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.info(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# --- Preprocessing ---
elif menu == "🛠️ Preprocessing":
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        st.title("🛠️ Data Preprocessing")

        # Missing Values
        st.subheader("🔍 Missing Values")
        st.write(df.isnull().sum())

        fill_method = st.radio("How do you want to fill missing values?",
                               ["Auto (Mean/Mode)", "Fill with 0", "Drop rows"],
                               horizontal=True)

        if fill_method == "Auto (Mean/Mode)":
            for col in df.columns:
                if df[col].dtype == object:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
        elif fill_method == "Fill with 0":
            df.fillna(0, inplace=True)
        else:
            df.dropna(inplace=True)

        st.success("✅ Missing values handled!")
        st.write(df.isnull().sum())

        # Encoding
        st.subheader("🔑 Encoding Categorical Variables")
        label_encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        st.dataframe(df.head(10))
        st.session_state.df = df

        # Download Button for Preprocessed Data
        st.subheader("⬇️ Download Preprocessed Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Preprocessed CSV",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv",
        )

    else:
        st.warning("⚠️ Please upload a dataset first in '📂 Upload Data'.")

# --- Outlier Handling ---
elif menu == "📉 Outlier Handling":
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        st.title("📉 Outlier Detection & Treatment")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select a numeric column:", numeric_cols)

            # Before
            st.subheader(f"📊 Boxplot for {col} (Before)")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

            # Outlier removal
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

            # After
            st.subheader(f"📊 Boxplot for {col} (After)")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[col], ax=ax2, color="orange")
            st.pyplot(fig2)

            st.success(f"✅ Removed outliers from '{col}'. New shape: {df.shape}")
            st.dataframe(df.head(10))
            st.session_state.df = df

            # Download Button for Cleaned Data
            st.subheader("⬇️ Download Data After Outlier Removal")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )

        else:
            st.warning("⚠️ No numeric columns available for outlier detection.")
    else:
        st.warning("⚠️ Please upload a dataset first in '📂 Upload Data'.")

# --- Modeling ---
elif menu == "🤖 Modeling":
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        st.title("🤖 Machine Learning Model")

        target_col = st.selectbox("Select Target Column", df.columns)

        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            st.write("✅ Features:", X.columns.tolist())

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            st.subheader("📈 Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
            with col2:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

            # Predictions vs Actual
            st.subheader("🔍 Predictions vs Actual")
            results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).reset_index(drop=True)
            st.dataframe(results.head(10))

            # Visualization
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Predicted vs Actual")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload and preprocess your dataset first.")
