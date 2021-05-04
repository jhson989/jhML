
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from functions import *


def square(input):
    return Square()(input)

def exp(input):
    return Exp()(input)



