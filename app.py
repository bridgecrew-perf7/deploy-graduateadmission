from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model_file = open('DecisionTreemodel.sav', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', chanceOfAdmit=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the admission chance based on 
    GRE Scores ( out of 340 )
    TOEFL Scores ( out of 120 )
    University Rating ( out of 5 )
    Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
    Undergraduate GPA ( out of 10 )
    Research Experience ( either 0 or 1 )
    '''
    GRE, TOEFL, UnivRating, SOP, LOR, CGPA, Research = [x for x in request.form.values()]

    data = []

    data.append(int(GRE))
    data.append(int(TOEFL))
    data.append(int(UnivRating))
    data.append(float(SOP))
    data.append(float(LOR))
    data.append(float(CGPA))

    if Research == "Yes":
        data.append(1)
    else:
        data.append(0)

    
    prediction = model.predict([data])
    output = prediction[0]

    return render_template('result.html', chanceOfAdmit=output, GRE=GRE, TOEFL=TOEFL, UnivRating=UnivRating, 
    SOP=SOP, LOR=LOR, CGPA=CGPA, Research=Research)


if __name__ == '__main__':
    app.run(debug=True)