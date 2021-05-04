import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from core.functions import *


def square(input):
    return Square()(input)

def exp(input):
    return Exp()(input)



