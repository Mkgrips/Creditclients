
import pandas as pd
import numpy as np
import pickle
import streamlit as st

def show_modele():
    

    st.title("MODELE Client Scoring Application")

    
    # Function to load pickle files
    def load_pickle(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    merged_avec_predictions= pd.read_csv("../data/data_merged_avec_predictionKNNetLGBM.csv")
    clf_lgbm = load_pickle('../data/clf_lgbm.pkl')
    data_client = load_pickle('../data/data_client.pkl')
    calibrated_lgbm=load_pickle('../data/calibrated_lgbm.pkl')
    # Load the merged_avec_predictions for client ID selection
    merged_avec_predictions= pd.read_csv("../data/data_merged_avec_predictionKNNetLGBM.csv")
    # Scoring function (modify based on your model and data structure)
    def score_client(client_data):
        # Reshape the client_data into a 2D array with one row
        client_data_2d = np.reshape(client_data, (1, -1))
        return clf_lgbm.predict_proba(client_data_2d)[:, 1]
    def score_client_calibrated(client_data):
        # Reshape the client_data into a 2D array with one row
        client_data_2d = np.reshape(client_data, (1, -1))
        return calibrated_lgbm.predict_proba(client_data_2d)[:, 1]


    # Select a client
    client_id = st.selectbox("Sélectionnez l'ID du client", 
                             merged_avec_predictions['SK_ID_CURR'].unique(), 
                             format_func=lambda x: str(x), 
                             key="client_id_selectbox")

    selected_client = merged_avec_predictions[merged_avec_predictions['SK_ID_CURR'] == client_id].iloc[0]

    # Display client score
    if st.button("Score Client"):
        # Assuming client_id is the index or a key in data_client to get the corresponding client data
        client_index = merged_avec_predictions.index[merged_avec_predictions['SK_ID_CURR'] == client_id].tolist()

        # Check if client_index is not empty
        if client_index:
            client_index = client_index[0]
            client_data = data_client[client_index]
            score = score_client(client_data)
            score_client_calibrated=score_client_calibrated(client_data)
            # Interpretation of the score
            if score < 0.3:
                interpretation = "Risque Élevé"
            elif score < 0.7:
                interpretation = "Risque Modéré"
            else:
                interpretation = "Faible Risque"
             # Interpretation of the score
            if score_client_calibrated < 0.3:
                interpretation_calibrated = "Risque Élevé"
            elif score_client_calibrated < 0.7:
                interpretation_calibrated = "Risque Modéré"
            else:
                interpretation_calibrated = "Faible Risque"
            
            # Display the score and its interpretation
            st.write("Valeur du Score Client:", score)
            st.write("Interprétation du Risque:", interpretation)
            st.write("Valeur du Score Client Calibré:", score_client_calibrated)
            st.write("Interprétation du Risque après Calibration:", interpretation_calibrated)





    # Get data for the selected client
    selected_client = merged_avec_predictions[merged_avec_predictions['SK_ID_CURR'] == client_id].iloc[0]

    # Display specific predicted probabilities for the selected client
    if st.button('Show predicted_probabilities_merge_LGBM_0'):
        st.write('predicted_probabilities_merge_LGBM_0:', selected_client['predicted_probabilities_merge_LGBM_0'])

    if st.button('Show predicted_probabilities_merge_LGBM_1'):
        st.write('predicted_probabilities_merge_LGBM_1:', selected_client['predicted_probabilities_merge_LGBM_1'])

    if st.button('Show predicted_probabilities_merge_0'):
        st.write('predicted_probabilities_merge_0:', selected_client['predicted_probabilities_merge_0'])

    if st.button('Show predicted_probabilities_merge_1'):
        st.write('predicted_probabilities_merge_1:', selected_client['predicted_probabilities_merge_1'])


def main():
    show_modele()
if __name__ == "__main__":
    main()
