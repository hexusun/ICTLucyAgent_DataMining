"""
Copyright (c) 2015. Docentron PTY LTD. All rights reserved.
This material may not be reproduced, displayed, modified or distributed without
the express prior written permission of the copyright holder.

Demo of loading a saved classification model and using it to predict a sentence
"""
__author__ = 'Docentron PTY LTD'

from lucy import text

from sklearn import svm
import pickle
clf = pickle.load(open("models/data_questions_model.p", "rb"))

# generate feature from a sentence
r = text.readvocab("models/data_questions_vocab.txt")
vocab = r[0]
vocabtf = r[1]
vocabidf = r[2]
#print vocab

fv = text.genfeatureVectorFromString("Is DES available in hardware?", vocab, vocabidf)
#print sum(fv)
r = clf.predict([fv])        # prediction (0 or 1)
p = clf.predict_proba([fv])  # probability of each of the classes
print r
print p
