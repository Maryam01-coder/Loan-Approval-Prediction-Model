import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px


#page set up
st.set_page_config(
    page_title="Loan Approval Prediction Model",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("cleaned_loan_approval_dataset.csv")
        return df
    except FileNotFoundError as e:
        st.warning(f"An Error occured: {e}")

@st.cache_resource
def load_model_artifact():
  try:
    model = joblib.load("loan_approval_prediction.pkl")
    rmc_model = joblib.load("rmc_loan_approval_prediction.pkl")
    feature = joblib.load("loan_approval_featured_columns.pkl")
    
    return model, rmc_model, feature
  except FileNotFoundError as e:
    st.error(f"Model artifact not found: {e}")
    st.stop()
  except Exception as e:
    st.error(f"An error occured: {e}")
    st.stop()


# =========================
# PAGE FUNCTIONS
# =========================
def format_number(value):
    if value >= 1000000:
        return f"{value / 1000000:.1f}M"
    elif value >= 1000:
        return f"{value / 1000:.1f}K"
    else:
        return f"{value:,.0f}"

def nav():
    pages = [
            "Overview",
            "Loan Data Visualization",
            "Model Interface - Logistic Regression",
            "Model Interface - Random Forest Classifier"
    ]
    
    selected = st.sidebar.selectbox(
        "All Pages",
        pages,
        index=pages.index(st.session_state.page),
        key="overview_nav"
        )
    
    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()

def show_overview(df):
    nav()
    st.title("Loan Data Analysis Visualization & Model Interface")
    st.write("A comprehensive analysis of loan records to predict loan approval.")

    st.markdown("---")
    
    #kpis
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,.0f}")

    with col2:
        educated = (df["education"] == "Graduate").sum()
        st.metric("Educated Candidates", f"{educated:,.0f}")

    with col3:
        avg_loan_term = df["loan_term"].mean()
        st.metric("Avg Loan Term", format_number(avg_loan_term))

    with col4:
        avg_loan_amount = df["loan_amount"].mean()
        st.metric("Avg Loan Amount", format_number(avg_loan_amount))

    
    #tables for df
    st.header("Dataset Overview")
    st.dataframe(df, width="content")

def show_dashboard(df):
    nav()
    st.title("Loan Data Analysis Visualization")
    st.write("Visualization of the indepth analysis of the loan records.")
    st.markdown("---")

    # Customer-specific filters
    st.sidebar.header("Filters")

    education = st.sidebar.multiselect(
        "Select Education",
        options=df["education"].unique(),
        default=df["education"].unique(),
    )

    self_employed = st.sidebar.multiselect(
        "Select Self-Employed",
        options=df["self_employed"].unique(),
        default=df["self_employed"].unique(),
    )
    
    def filter_data(df, education, self_employed):
        filtered_df = df[
            (df["education"].isin(education)) & 
            (df["education"].isin(education)) &
            (df["self_employed"].isin(self_employed)) & 
            (df["self_employed"].isin(self_employed))
        ]
        return filtered_df
    
    filtered_df = filter_data(df, education, self_employed)

    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,.0f}")

    with col2:
        total_graduate = (filtered_df["education"] == "Graduate").sum() if len(filtered_df) > 0 else 0
        st.metric("Educated Candidates", f"{total_graduate:,.0f}")

    with col3:
        total_self_emplyed = (filtered_df["self_employed"] == "Yes").sum() if len(filtered_df) > 0 else 0
        st.metric("Self-Employed Candidates", f"{total_self_emplyed:,.0f}")

    with col4:
        avg_credit_score = filtered_df["credit_score"].mean() if len(filtered_df) > 0 else 0
        st.metric("Average Credit Score", format_number(avg_credit_score))

    # charts
    if len(filtered_df) == 0:
        st.warning('No Filter Selected. Please Adjust Your Selection.')
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Distribution of Educated Candidates')
        fig = px.pie(
            names=filtered_df["education"].unique(),
            values=filtered_df["education"].value_counts()
        )

        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader('Distribution of Employed Candidates')
        items = filtered_df["self_employed"].value_counts()
        fig1 = px.pie(
            names=items.index,
            values=items.values,
        )
        st.plotly_chart(fig1, width='stretch')

    st.subheader('Distribution of Approved Loans')
    fig2 = px.pie(
        names=filtered_df["loan_status"].unique(),
        values=filtered_df["loan_status"].value_counts()
    )
    st.plotly_chart(fig2, width='stretch') 


    st.subheader('Distribution of Income of Candidates')

    fig3 = px.histogram(
        df,
        x="income_annum",
        nbins=10,
        height=500
    )
    fig3.update_traces(
        marker_line_color="white",
        marker_line_width=1
    )

    st.plotly_chart(fig3, width='stretch') 

    st.subheader('Correlation Between Loan Amount and Yearly Income')
    fig0 = px.scatter(
        df, 
        x=df["income_annum"], 
        y=df["loan_amount"],
        height=600
    )
    st.plotly_chart(fig0, width='stretch') 

    st.subheader('Correlation Between All Numerical Columns')

    correlation = df[["no_of_dependents", "loan_term", "credit_score", "income_annum", "loan_amount", 
                  "residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]].corr()

    fig5 = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",   
        width=800,                           
        height=600 
    )
    st.plotly_chart(fig5, width='stretch')
    

def show_interface(df, model, feature):
    nav()

    st.title("Loan Approval Prediction")
    st.write("Enter the applicant's information to predict loan approval.")

    # ==========================================
    # INITIALIZE SESSION STATE
    # ==========================================

    if "logistic_prediction" not in st.session_state:
        st.session_state.logistic_prediction = None

    if "logistic_probability" not in st.session_state:
        st.session_state.logistic_probability = None

    if "logistic_candidate_data" not in st.session_state:
        st.session_state.logistic_candidate_data = None

    if "logistic_show_result" not in st.session_state:
        st.session_state.logistic_show_result = False

    # Numerical inputs
    input_defaults = {
        "no_of_dependents": 0,
        "income_annum": 0,
        "loan_amount": 0,
        "loan_term": 0,
        "credit_score": 0,
        "residential_assets_value": 0,
        "commercial_assets_value": 0,
        "luxury_assets_value": 0,
        "bank_asset_value": 0
    }

    for key, value in input_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Categorical inputs
    if "education" not in st.session_state:
        st.session_state.education = "Graduate"

    if "self_employed" not in st.session_state:
        st.session_state.self_employed = "Yes"

    st.divider()

    # ==========================================
    # INPUT FORM
    # ==========================================

    st.markdown("### Applicant Information")
    st.caption("Enter the applicant's personal and financial details below.")
    
    with st.form("logistic_regression_form"):
        # Numerical features
        st.markdown("#### Financial and Personal Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            no_of_dependents = st.number_input(
                "Number of Dependents",
                min_value=0,
                value=0,
                step=1,
                key="no_of_dependents"
            )

            income_annum = st.number_input(
                "Annual Income",
                min_value=0,
                value=0,
                step=100000,
                format="%d",
                key="income_annum"
            )

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0,
                value=int(df["loan_amount"].median()),
                step=100000,
                format="%d",
                key="loan_amount"
            )

        with col2:
            loan_term = st.number_input(
                "Loan Term",
                min_value=1,
                value=int(df["loan_term"].median()),
                step=1,
                key="loan_term"
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=0,
                value=int(df["credit_score"].median()),
                step=10,
                key="credit_score"
            )

            residential_assets_value = st.number_input(
                "Residential Assets Value",
                min_value=0,
                value=int(df["residential_assets_value"].median()),
                step=100000,
                format="%d",
                key="residential_assets_value"
            )

        with col3:
            commercial_assets_value = st.number_input(
                "Commercial Assets Value",
                min_value=0,
                value=int(df["commercial_assets_value"].median()),
                step=100000,
                format="%d",
                key="commercial_assets_value"
            )

            luxury_assets_value = st.number_input(
                "Luxury Assets Value",
                min_value=0,
                value=int(df["luxury_assets_value"].median()),
                step=100000,
                format="%d",
                key="luxury_assets_value"
            )

            bank_asset_value = st.number_input(
                "Bank Asset Value",
                min_value=0,
                value=int(df["bank_asset_value"].median()),
                step=100000,
                format="%d",
                key="bank_asset_value"
            )

        st.divider()

        # Categorical features
        st.markdown("#### Additional Information")

        col1, col2 = st.columns(2)

        with col1:
            education = st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"],
                key="education"
            )

        with col2:
            self_employed = st.selectbox(
                "Self Employed",
                ["Yes", "No"],
                key="self_employed"
            )

        st.divider()

        # Submit button
        predict_button = st.form_submit_button(
            "Predict Loan Approval",
            width="stretch"
        )

    # ==========================================
    # MAKE PREDICTION
    # ==========================================

    if predict_button:

        candidate_data = {
            "no_of_dependents": no_of_dependents,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "credit_score": credit_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
            "education": education,
            "self_employed": self_employed
        }

        # Convert input into DataFrame
        customer_df = pd.DataFrame(
            [candidate_data],
            columns=feature
        )

        # Model prediction
        prediction = model.predict(customer_df)[0]
        probability = model.predict_proba(customer_df)[0]

        # Clean class labels
        classes = [str(label).strip() for label in model.classes_]

        prediction_result = str(prediction).strip()

        approved_probability = probability[classes.index("Approved")]

        rejected_probability = probability[classes.index("Rejected")]

        # Store results in session state
        st.session_state.logistic_prediction = prediction_result

        st.session_state.logistic_probability = {
            "approved": approved_probability,
            "rejected": rejected_probability
        }

        st.session_state.logistic_candidate_data = candidate_data

        st.session_state.logistic_show_result = True

    # ==========================================
    # DISPLAY PREDICTION RESULT
    # ==========================================

    if st.session_state.logistic_show_result:

        st.divider()

        st.subheader("Prediction Result")

        prediction_result = st.session_state.logistic_prediction

        probability = st.session_state.logistic_probability

        if prediction_result == "Approved":

            st.success("Loan Approved!")

        else:

            st.error("Loan Rejected!")

        # Display probabilities
        st.write("### Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Approval Probability",
                f"{probability['approved'] * 100:.2f}%"
            )

            st.progress(float(probability["approved"]))

        with col2:
            st.metric(
                "Rejection Probability",
                f"{probability['rejected'] * 100:.2f}%"
            )

            st.progress(float(probability["rejected"]))

        # ==========================================
        # CLEAR PREDICTION
        # ==========================================

        if st.button("Clear Prediction", key="lr_clear"):

            st.session_state.logistic_prediction = None
            st.session_state.logistic_probability = None
            st.session_state.logistic_candidate_data = None
            st.session_state.logistic_show_result = False

            st.rerun()

def show_rmc_interface(df, rmc_model, feature):
    nav()
    st.title("Loan Approval Prediction")
    st.write("Enter the applicant's information to predict loan approval.")

    # ==========================================
    # INITIALIZE SESSION STATE
    # ==========================================

    if "random_forest_prediction" not in st.session_state:
        st.session_state.random_forest_prediction = None

    if "random_forest_probability" not in st.session_state:
        st.session_state.random_forest_probability = None

    if "random_forest_candidate_data" not in st.session_state:
        st.session_state.random_forest_candidate_data = None

    if "random_forest_show_result" not in st.session_state:
        st.session_state.random_forest_show_result = False

    # Numerical inputs
    input_defaults = {
        "rf_no_of_dependents": 0,
        "rf_income_annum": 0,
        "rf_loan_amount": 0,
        "rf_loan_term": 0,
        "rf_credit_score": 0,
        "rf_residential_assets_value": 0,
        "rf_commercial_assets_value": 0,
        "rf_luxury_assets_value": 0,
        "rf_bank_asset_value": 0
    }

    for key, value in input_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Categorical inputs
    if "rf_education" not in st.session_state:
        st.session_state.rf_education = "Graduate"

    if "rf_self_employed" not in st.session_state:
        st.session_state.rf_self_employed = "Yes"

    st.divider()

    # ==========================================
    # INPUT FORM
    # ==========================================

    st.markdown("### Applicant Information")
    st.caption("Enter the applicant's personal and financial details below.")

    with st.form("random_forest_form"):

        st.markdown("#### Financial and Personal Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            no_of_dependents = st.number_input(
                "Number of Dependents",
                min_value=0,
                value=0,
                step=1,
                key="rf_no_of_dependents"
            )

            income_annum = st.number_input(
                "Annual Income",
                min_value=0,
                value=0,
                step=100000,
                format="%d",
                key="rf_income_annum"
            )

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0,
                value=0,
                step=100000,
                format="%d",
                key="rf_loan_amount"
            )

        with col2:
            loan_term = st.number_input(
                "Loan Term",
                min_value=1,
                value=int(df["loan_term"].median()),
                step=1,
                key="rf_loan_term"
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=0,
                value=int(df["credit_score"].median()),
                step=10,
                key="rf_credit_score"
            )

            residential_assets_value = st.number_input(
                "Residential Assets Value",
                min_value=0,
                value=0,
                step=100000,
                format="%d",
                key="rf_residential_assets_value"
            )

        with col3:
            commercial_assets_value = st.number_input(
                "Commercial Assets Value",
                min_value=0,
                value=0,
                step=100000,
                format="%d",
                key="rf_commercial_assets_value"
            )

            luxury_assets_value = st.number_input(
                "Luxury Assets Value",
                min_value=0,
                value=0,
                step=100000,
                format="%d",
                key="rf_luxury_assets_value"
            )

            bank_asset_value = st.number_input(
                "Bank Asset Value",
                min_value=0,
                value=0,
                step=100000,
                format="%d",
                key="rf_bank_asset_value"
            )

        st.divider()

        # Categorical features
        st.markdown("#### Additional Information")

        col1, col2 = st.columns(2)

        with col1:
            education = st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"],
                key="rf_education"
            )

        with col2:
            self_employed = st.selectbox(
                "Self Employed",
                ["Yes", "No"],
                key="rf_self_employed"
            )

        st.divider()

        predict_button = st.form_submit_button(
            "Predict Loan Approval",
            width="stretch"
        )

    # ==========================================
    # MAKE PREDICTION
    # ==========================================

    if predict_button:

        candidate_data = {
            "no_of_dependents": no_of_dependents,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "credit_score": credit_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
            "education": education,
            "self_employed": self_employed
        }

        # Convert input into DataFrame
        customer_df = pd.DataFrame(
            [candidate_data],
            columns=feature
        )

        # Model prediction
        prediction = rmc_model.predict(customer_df)[0]
        probability = rmc_model.predict_proba(customer_df)[0]

        # Clean class labels
        classes = [str(label).strip() for label in rmc_model.classes_]

        prediction_result = str(prediction).strip()

        approved_probability = probability[classes.index("Approved")]
        rejected_probability = probability[classes.index("Rejected")]

        # Store results in session state
        st.session_state.random_forest_prediction = prediction_result

        st.session_state.random_forest_probability = {
            "approved": approved_probability,
            "rejected": rejected_probability
        }

        st.session_state.random_forest_candidate_data = candidate_data

        st.session_state.random_forest_show_result = True

    # ==========================================
    # DISPLAY PREDICTION RESULT
    # ==========================================

    if st.session_state.random_forest_show_result:

        st.divider()

        st.subheader("Prediction Result")

        prediction_result = st.session_state.random_forest_prediction

        probability = st.session_state.random_forest_probability

        if prediction_result == "Approved":
            st.success("Loan Approved!")
        else:
            st.error("Loan Rejected!")

        # Display probabilities
        st.write("### Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Approval Probability",
                f"{probability['approved'] * 100:.2f}%"
            )

            st.progress(float(probability["approved"]))

        with col2:
            st.metric(
                "Rejection Probability",
                f"{probability['rejected'] * 100:.2f}%"
            )

            st.progress(float(probability["rejected"]))

        # ==========================================
        # CLEAR PREDICTION
        # ==========================================

        if st.button("Clear Prediction", key="rf_clear"):

            st.session_state.random_forest_prediction = None
            st.session_state.random_forest_probability = None
            st.session_state.random_forest_candidate_data = None
            st.session_state.random_forest_show_result = False

            st.rerun()

# =========================
# MAIN FUNCTION
# =========================

def main():
    df = load_dataset()
    model, rmc_model, feature = load_model_artifact()

    # Initialize current page
    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    #Page routing
    if st.session_state.page == "Overview":
        show_overview(df)

    elif st.session_state.page == "Loan Data Visualization":
        show_dashboard(df)

    elif st.session_state.page == "Model Interface - Logistic Regression":
        show_interface(df, model, feature)

    elif st.session_state.page == "Model Interface - Random Forest Classifier":
        show_rmc_interface(df, rmc_model, feature)

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    main()