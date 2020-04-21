from flask import Flask

app = Flask(__name__)

from dog_breed_web_app import routes
