from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Temporary storage for survey questions.
# TODO: Replace this with a Postgres DB.
questions = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_survey', methods=['GET', 'POST'])
def create_survey():
    if request.method == 'POST':
        question = request.form['question']
        questions.append({'question': question})
        return redirect(url_for('create_survey'))
    return render_template('create_survey.html', questions=questions)

@app.route('/take_survey', methods=['GET', 'POST'])
def take_survey():
    if request.method == 'POST':
        # Process submitted answers here
        answers = request.form
        # For demo, just print answers
        print(answers)
        return redirect(url_for('index'))
    return render_template('take_survey.html', questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
