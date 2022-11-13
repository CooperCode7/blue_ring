"""This is just a placeholder app for establishing the dynos"""

import os
from flask import Flask

app = Flask(__name__)


@app.route("/")
def index():
    return "Champions of Winning, Superb!"


if __name__ == "__main__":
    app.run(os.getenv("PORT"))