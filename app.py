from flask import Flask, render_template, request
from python_script import translator

app = Flask(__name__)

@app.route('/', methods = ["GET" , "POST"])
def index():
    if request.method == 'POST':
      input_en = request.form['input_en']
      output_table, output_command = translator.main(input_en)
      return render_template('index.html', output_table=output_table, input_en=input_en, output_command=output_command)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 50000, debug=True)