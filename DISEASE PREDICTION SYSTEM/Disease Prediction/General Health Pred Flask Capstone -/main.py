from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np
from scipy.stats import mode
import pickle

app = Flask(__name__)



model = pickle.load(open('model.pkl', 'rb'))
model2=pickle.load(open('model2.pkl','rb'))
model3=pickle.load(open('model3.pkl','rb'))
model4=pickle.load(open('model4.pkl','rb'))
model5=pickle.load(open('model5.pkl','rb'))
model6=pickle.load(open('model6.pkl','rb'))
model7=pickle.load(open('model7.pkl','rb'))


symptoms_dict={'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'blood_in_sputum': 117, 'prominent_veins_on_calf': 118, 'palpitations': 119, 'painful_walking': 120, 'pus_filled_pimples': 121, 'blackheads': 122, 'scurring': 123, 'skin_peeling': 124, 'silver_like_dusting': 125, 'small_dents_in_nails': 126, 'inflammatory_nails': 127, 'blister': 128, 'red_sore_around_nose': 129, 'yellow_crust_ooze': 130}
symptoms_dict = pd.DataFrame(list(symptoms_dict.items()), columns= ['Symptoms', 'Count'])


@app.route("/")
def getModel():
    return render_template('index.html')



@app.route("/", methods=['GET','POST'])
def predict():
    input_vector = np.zeros(len(symptoms_dict))
    
    
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms_checkbox')
        if(len(symptoms)==0):
            return render_template('index.html')
        for i in range(0, len(symptoms)): 
            symptoms[i] = int(symptoms[i]) 
        for symptom in symptoms:
            input_vector[symptom] = 1 
            # symp = []
            # symp.append(symptoms_dict.iloc[symptom, 1])
            # input_vector[symp] = 1
            
        pred1 = model.predict([input_vector])[0]
        pred2 = model2.predict([input_vector])[0]
        pred3 = model3.predict([input_vector])[0]
        pred4 = model4.predict([input_vector])[0]
        pred5 = model5.predict([input_vector])[0]
        pred6 = model6.predict([input_vector])[0]
        pred7 = model7.predict([input_vector])[0]
        finalprediction=mode([pred1,pred2,pred3,pred4,pred5,pred6,pred7])[0][0]
        finalprediction=mode([pred1,pred2,pred4,pred5,pred6,pred7])[0][0]
        finalprediction=mode([pred1,pred4,pred6])[0][0]
        
        return render_template('show.html',symp1=input_vector, disease= finalprediction,d1=pred1,d2=pred2,d3=pred3,d4=pred4,d5=pred5,d6=pred6,d7=pred7)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="localhost",port=5002,debug=True)