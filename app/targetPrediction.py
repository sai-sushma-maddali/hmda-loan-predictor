import pickle

def predict_loan_decision(processed_data):
    # Load RF model

    X_one_pca = processed_data.copy()

    with open("best_rf_model.pkl", "rb") as file:
        rf = pickle.load(file)

    output_pred = rf.predict(X_one_pca)

    # Load le

    with open("target_label_encoder.pkl", "rb") as file:
        le = pickle.load(file)


    loan_decision = le.inverse_transform(rf.predict(X_one_pca))[0]

    return loan_decision