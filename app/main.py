import streamlit as st
import json
import dataPrep
import dataProcessing
import targetPrediction
import os
file_path = os.path.join(os.getcwd(), "app", "info.json")

with open(file_path, "r") as json_file:
    info = json.load(json_file)

st.title("Home Mortgage Disclosure Act (HMDA) Loan Decision Prediction")
st.write("This application predicts the loan decision based on various applicant and property information.")
agency_option = st.selectbox(
    "Agency name",
    info["agency_name_list"],
)
st.divider()

property_option = st.selectbox(
    "Property Type",
    info["property_type_name_list"],
)
st.divider()


owner_occupancy_option = st.selectbox(
    "Owner Occupancy",
    info["owner_occupancy_name_list"],
)
st.divider()


county_name_value = st.text_input(
    "County Name (e.g., Adams County)"
)

st.divider()

census_tract_value = st.text_input(
    "Census Tract Number"
)
st.divider()

# loan_amount_value = st.text_input(
#     "Loan Amount (000s)"
# )

loan_amount_value = st.number_input("Loan Amount (000s)", min_value=50, max_value=2000, step=10)

st.divider()


applicant_income_value = st.text_input(
    "Applicant Income (000s)"
)
st.divider()

preapproval_option = st.selectbox(
    "Preapproval name",
    info["preapproval_name_list"],
)

st.divider()

applicant_ethnicity_option = st.selectbox(
    "Applicant Ethnicity",
    info["applicant_ethnicity_name_list"],
)
st.divider()

co_applicant_ethnicity_option = st.selectbox(
    "Co-Applicant Ethnicity",
    info["co_applicant_ethnicity_name_list"],
)
st.divider()

applicant_race_option = st.selectbox(
    "Applicant Race",
    info["applicant_race_name_1_list"],
)
st.divider()

co_applicant_race_option = st.selectbox(
    "Applicant Race",
    info["co_applicant_race_name_1_list"],
)
st.divider()

applicant_gender_option = st.selectbox(
    "Applicant Gender",
    info["applicant_gender_list"],
)

st.divider()

co_applicant_gender_option = st.selectbox(
    "Co-Applicant Gender",
    info["co_applicant_gender_list"],
)
st.divider()

purchaser_type_option = st.selectbox(
    "Purchaser Type",
    info["purchaser_type_name_list"],
)
st.divider()

hoepa_option = st.selectbox(
    "Hoepa Status",
    info["hoepa_status_name_list"],
)
st.divider()

lien_status_option = st.selectbox(
    "Lien Status",
    info["lien_status_name_list"]
)


# Gathering data
sample_data = {
    "agency_name": agency_option,
    "property_type_name": property_option,
    "owner_occupancy_name": owner_occupancy_option,
    "loan_amount_000s": loan_amount_value,
    "preapproval_name": preapproval_option,
    "county_name": county_name_value,
    "census_tract_number": census_tract_value,  
    "applicant_ethnicity_name": applicant_ethnicity_option,
    "co_applicant_ethnicity_name": co_applicant_ethnicity_option,
    "applicant_race_name_1": applicant_race_option,
    "co_applicant_race_name_1": co_applicant_race_option,
    "applicant_sex_name": applicant_gender_option,
    "co_applicant_sex_name": co_applicant_gender_option,
    "purchaser_type_name": purchaser_type_option,
    "hoepa_status_name": hoepa_option,
    "lien_status_name": lien_status_option,
    "applicant_income_000s": applicant_income_value
}

# st.write(sample_data)

@st.dialog("Loan Decision")
def vote(loan_decision):
    st.write(f"{loan_decision}")

clicked = st.button("Predict")
if clicked:
    # Prepare data
    prepared_data = dataPrep.prepare_data(sample_data)

    # Process data
    processed_data = dataProcessing.process_data(prepared_data)

    # Predict loan decision
    loan_decision = targetPrediction.predict_loan_decision(processed_data)
    # st.write(f"Loan Decision: {loan_decision}")
    vote(loan_decision)

