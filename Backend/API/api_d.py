# import string
from flask_cors import CORS, cross_origin
# from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
from flask_mail import Mail,Message
# import smtplib, ssl

import model.diabetes_model as diabetesModel
import model.parkinson_model as parkinsonModel
import model.heart_model as heartModel

app = Flask(__name__)
CORS(app)

# app.config['SECRET_KEY'] = "tsfyguaistyatuis589566875623568956"
app.config['MAIL_SERVER'] = "smtp.googlemail.com"
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = "kenilkalathiya97@gmail.com"
app.config['MAIL_PASSWORD'] = "ayzefwzdrxpqpknu"




@app.route('/heartDisease', methods=['POST'])
def heartDisease():
    content = request.json
    print(content)
    input_data =float(content['one']),float(content['two']),float(content['three']),float(content['four']),float(content['five']),float(content['six']),float(content['seven']),float(content['eight']),float(content['nine']),float(content['ten']),float(content['eleven']),float(content['twelve']),float(content['thirteen'])
    output = heartModel.heartModelFunc(input_data)
    op = int(output)
    return jsonify({'prediction' : op})


@app.route('/diabetesDisease', methods=['POST'])
def diabetesDisease():
    content = request.json
    print(content)
    input_data =float(content['one']),float(content['two']),float(content['three']),float(content['four']),float(content['five']),float(content['six']),float(content['seven']),float(content['eight'])
    output = diabetesModel.detectDiabetesFunc(input_data)
    op = int(output)
    return jsonify({'prediction' : op})


@app.route('/parkinsonDisease', methods=['POST'])
def parkinsonDisease():
    content = request.json
    print(content)
    input_data =(content['one']),(content['two']),(content['three']),(content['four']),(content['five']),(content['six']),(content['seven']),(content['eight']),(content['nine']),(content['ten']),(content['eleven']),(content['twelve']),(content['thirteen']), (content['fourteen']), (content['fifteen']), (content['sixteen']), (content['seventeen']), (content['eighteen']), (content['nineteen']), (content['twenty']), (content['TO']), (content['TT'])
    output = parkinsonModel.parkinsonModelFunc(input_data)

    op = int(output)
    return jsonify({'prediction' : op})

mail = Mail(app)
@app.route('/contactUs', methods=['POST'])
def contactUs():
    content = request.json
    if request.method == "POST":
        name = content["one"]
        Usermail = content["two"]
        message = content["three"]

    msg = Message("This is Contact Us Form", sender=Usermail, recipients=['kenilkalathiya73@gmail.com'])
    msg.body = f"Name : {name} \n Mail : {Usermail} \n Query : {message}"

    try:

        mail.send(msg)
        return jsonify({'status' : 1})
    except Exception as e:
        print(e)
        return jsonify({'status' : 0})


if __name__ == "__main__":
    app.run(debug=True)