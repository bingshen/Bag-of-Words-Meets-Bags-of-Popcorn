from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
import os
import numpy as np
import random

class BasicLSTM(object):
    def __init__(self,iterator,src_vocab_table):
        self.iterator=iterator
        self.src_vocab_table=src_vocab_table