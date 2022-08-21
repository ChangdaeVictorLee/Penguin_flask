## sklearn 버전이 가상환경과 모델을 만든 곳이 다를 경우 에러가 뜸
## sklearn을 업그레이드 하는 것이 좋음

from flask import Flask, render_template, request, jsonify
import sys
import traceback
import joblib
import numpy as np

application = Flask(__name__)

model_file_name = 'penguin.pkl'


@application.route("/", methods=['GET','POST'])
def hello():
    try:    
        if request.method == 'GET':
            return render_template('index.html')
        else:            
            param = [[]]
            param[0].append(float(request.form['bill_len']))
            param[0].append(float(request.form['bill_depth']))
            param[0].append(float(request.form['flipper']))
            param[0].append(float(request.form['body_mass']))
            
            img_name = ['Adelie.jpg', 'Chinstrap.jpg', 'Gentoo.jpg']

            
            prediction = list(joblib.load(model_file_name).predict(param))

            return render_template('index.html', prediction = prediction, img=img_name[prediction[0]])

    
    except Exception as e:
        return jsonify({"error": str(e), 'trace': traceback.format_exc()})
        

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=int(sys.argv[1]))
