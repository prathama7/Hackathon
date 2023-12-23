import pickle
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model_path = 'random_forest_model.pkl'
model = pickle.load(open(model_path, 'rb'))

# Define the function to preprocess input data
def preprocess_data(Curvature, Flow, NDWI, Precipitation, Slope):
    # Ensure non-negative values
    Curvature = max(Curvature, 0)
    Flow = max(Flow, 0)
    NDWI = max(NDWI, 0)
    Precipitation = max(Precipitation, 0)
    Slope = max(Slope, 0)

    # Create a dataframe with the input data
    data = {'Curvature': [Curvature], 'Flow': [Flow], 'NDWI': [NDWI], 'Precipitation': [Precipitation], 'Slope': [Slope]}
    df = pd.DataFrame(data)

    return df

# Define the function to predict landslide
def predict_landslide(Curvature, Flow, NDWI, Precipitation, Slope):
    # Preprocess the input data
    df = preprocess_data(Curvature, Flow, NDWI, Precipitation, Slope)

    # Predict landslide using the trained model
    probability_of_risk = model.predict_proba(df)[:, 1]

    # Return the probability of risk
    return probability_of_risk[0]

# Create the Streamlit app
def main():
    # Set the title and description
    st.header("Pahiro AI")
    st.write("Many places in Nepal face a growing problem of landslides and have to face the various side effects caused by them.")
    st.write("Alert yourself and your loved ones about possible landslides around your area!")
    st.image("landslide.jpg", caption="Landslide in prone regions", use_column_width=True)
    st.write("The Mugling–Narayanghat road is located at central Nepal. The road connects not only Mugling and Narayanghat towns but also eastern and western part of country.The road passes through plain, rolling and hilly terrain with curved alignment.The road passes through plain rolling terrain with curved alignment. The area receives average annual rainfall of about 2400 mm and the area belongs to subtropical zone.")
    st.write("The road passes by cut slopes of Mahabharat range of rolling and hilly terrain. According to Joshi, (2006); the geological units along the road sections are of following types and the distribution pattern: types and the distribution pattern Kuncha formation of highly weathered phyllite and metasandstone,Fagfog quartzite, grey feeble dandagaon phyllites. Norpul formation, which extensively distributed and comprises purebesi quartzite,and slates, Grey dhading dolomite. Cleaved carbonaceous benighat slates and Dun valley gravel.")

    # Add a divider
    st.markdown("<hr>", unsafe_allow_html=True)

    st.sidebar.header("Enter the Required Information")

    with st.sidebar.form("user_input_form"):
        Curvature = st.slider("Curvature", min_value=1, max_value=5, step=1)
        Slope = st.slider("Slope", min_value=1, max_value=5, step=1)
        Precipitation = st.slider("Precipitation", min_value=1, max_value=5, step=1)
        NDWI = st.slider("NDWI", min_value=1, max_value=5, step=1)
        Flow = st.slider("Flow", min_value=1, max_value=5, step=1)

        detect_button = st.form_submit_button("Detect")
        if detect_button:
            if Curvature <= 5 and Slope <= 5 and Precipitation <= 5 and NDWI <= 5 and Flow <= 5:
                probability_of_risk = predict_landslide(Curvature, Flow, NDWI, Precipitation, Slope)

            # Map the prediction to the corresponding label
                prediction_label = 'landslide.' if probability_of_risk > 0.5 else 'landslide.'

            # Display the prediction and probability
                st.write(f"There is {probability_of_risk*100:.2f}% probability of", prediction_label)
            else:
                st.warning("Please enter valid values.")                

if __name__ == '__main__':
    main()
