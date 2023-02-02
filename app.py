from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server
import pandas as pd
import pickle
import numpy as np
import fasttext as ft

app = Flask(__name__)
def pred():
    inp = input('Enter text',type='text')
    model = ft.load_model('yo.bin')
    out = model.predict([inp])
    put_html(f'<h1>{out[0][0][0][9:]}</h1>')

if __name__ == '__main__':


    start_server(pred ,port=8000,debug=True)