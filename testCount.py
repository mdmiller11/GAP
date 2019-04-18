# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:46:34 2019

@author: O345
"""
import re

def count_occurrences(word, sentence):
    return re.sub('[^0-9A-Za-z]',' ',sentence).lower().split().count(word)

text = "Upon their acceptance into the Kontinental Hockey League, Dehner left Finland to sign a contract in Germany with EHC M*nchen of the DEL on June 18, 2014. After capturing the German championship with the M*nchen team in 2016, he left the club and was picked up by fellow DEL side EHC Wolfsburg in July 2016. Former NHLer Gary Suter and Olympic-medalist Bob Suter are Dehner's uncles. His cousin is Minnesota Wild's alternate captain Ryan Suter."
word = "Bob Suter"

print(count_occurrences(word, text))
print(text.count(word))
