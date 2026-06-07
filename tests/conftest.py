import sys
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
prototype = os.path.join(base, 'prototype')

sys.path.insert(0, base)

sys.path.insert(0, prototype)