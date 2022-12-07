from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
from scipy.stats import mode
import statistics

app = Flask(__name__)

# model = pickle.load(open('model.pkl', 'rb'))
model2=pickle.load(open('model2.pkl','rb'))
# model3=pickle.load(open('model3.pkl','rb'))
# model4=pickle.load(open('model4.pkl','rb'))
# model5=pickle.load(open('model5.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    # prediction = model.predict(final)[0]
    prediction1 = model2.predict(final)[0]
    # prediction2 = model3.predict(final)[0]
    # prediction3 = model4.predict(final)[0]
    # prediction4 = model5.predict(final)[0]


    # prob=model.predict_proba(final)[0][1]
    prob1=model2.predict_proba(final)[0][1]
    # prob2=model3.predict_proba(final)[0][1]
    # prob3=model4.predict_proba(final)[0][1]
    # prob4=model5.predict_proba(final)[0][1]

    finalprediction=mode([prediction1])[0][0]
    finalprob=statistics.mean([prob1])

    if finalprediction == 1:
        return render_template('index.html', pred='You are tested positive and your probability of coronavirus is :'.format(finalprediction),inf=finalprob*100,save='Save Your Result!')
    else:
        return render_template('index.html', pred='You are tested negative and your probability of coronavirus is :'.format(finalprediction),inf=finalprob*100,save='Save Your Result!')


if __name__ == '__main__':
    app.run(host="localhost",port=5001,debug=True)
