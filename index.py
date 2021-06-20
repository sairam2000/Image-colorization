from flask import Flask, render_template, request
import os
import sys
import object_detection
import base64

app = Flask(__name__)

sys.stdout = open('output.txt','w')


@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")


@app.route('/obj_detector')
def object_detector():
    return render_template('object_detector.html')


@app.route('/img_colorizer')
def img_colorizer():
    return render_template('img_colorizer.html')


@app.route('/detect', methods=['POST'])
def detect():
    f = request.files['img']
    f.save('uploads/'+f.filename)
    with open('uploads/'+f.filename, "rb") as img_file:
        input_file = base64.b64encode(img_file.read())
    input_file = input_file.decode('utf-8')
    object_detection.predict('uploads/'+f.filename)
    with open('./images/result.png', "rb") as img_file:
        output_file = base64.b64encode(img_file.read())
    output_file = output_file.decode('utf-8')
    input_file = "data:image/jpeg;base64, "+input_file
    output_file = "data:image/jpeg;base64, "+output_file
    os.remove('uploads/'+f.filename)
    os.remove('./images/result.png')
    return render_template('object_detector.html', result=True, input_file=input_file, output_file=output_file)


if __name__ == "__main__":
    app.run()
