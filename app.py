from flask import Flask, render_template, request
from main import create_vector_db, get_answer_chain

app = Flask(__name__)

# Provide the correct path to your PDF document
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        file_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(file_path)

        create_vector_db(file_path)
        chain = get_answer_chain()

        return render_template('index.html', success='File uploaded successfully!', chain=chain)

    return render_template('index.html', error='Invalid file format')


@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')

    if not question:
        return render_template('index.html', error='Please enter a question')

    chain = get_answer_chain()
    answer = chain(question)
    return render_template('index.html', question=question, answer=answer)


if __name__ == '__main__':
    app.run(debug=True)
