import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
classifier_dt = pickle.load(open('Placementpredictdt.pkl','rb'))
classifier_knn = pickle.load(open('Placementpredictknn.pkl','rb'))
classifier_svm = pickle.load(open('Placementpredictsvm.pkl','rb'))
classifier_rf = pickle.load(open('Placementpredictrf.pkl','rb'))


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    
    '''
    tenth = int(request.args.get('tenth'))
    twelfth = int(request.args.get('twelfth'))
    btech = int(request.args.get('btech'))
    sem7 = int(request.args.get('sem7'))
    sem6 = int(request.args.get('sem6'))
    sem5 = int(request.args.get('sem5'))
    final = int(request.args.get('sem5'))
    medium = int(request.args.get('medium'))
    Model = (request.args.get('Model'))

    
    if Model=="Decision Tree Classifier":
        prediction = classifier_dt.predict([[tenth,twelfth,btech,sem7,sem6,sem5,final,medium]])
    
    elif Model=="KNN Classifier":
        prediction = classifier_knn.predict([[tenth,twelfth,btech,sem7,sem6,sem5,final,medium]])
    
    elif Model=="SVM Classifier":
        prediction = classifier_svm.predict([[tenth,twelfth,btech,sem7,sem6,sem5,final,medium]])
    
    else:
        prediction = classifier_rf.predict([[tenth,twelfth,btech,sem7,sem6,sem5,final,medium]])
    
    print("Survived", prediction)
    if prediction==[1]:
        prediction="The student got placed"
    else:
        prediction="The student is not placed"
    print(prediction)
        
    return render_template('index.html', prediction_text='Classification Model has predicted the survival based on various parameters: {}'.format(prediction))
    
  
if __name__=="__main__":
    app.run(debug=True)


