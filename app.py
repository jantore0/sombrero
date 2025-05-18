from flask import Flask, send_file

app = Flask(__name__)

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/final_highlights.mp4')
def get_highlights():
    return send_file('final_highlights.mp4')

if __name__ == '__main__':
    app.run(debug=True)