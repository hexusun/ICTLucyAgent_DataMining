"""
Copyright (c) 2015. Docentron PTY LTD. All rights reserved.
This material may not be reproduced, displayed, modified or distributed without
the express prior written permission of the copyright holder.

Lucy message processing using a usenet question model classifier
Uses the classification model trained on the usenet questions to classify sentences.

This sample illustrates how to use the API to get messages received and send a reply to the sender for a chapter you created.

First thing to do:
    You need to first create an account at https:/www.a.kopo.com/wp/mes
    Next, create a journey using the designer. Click designer after login and add a journey.
    Next, create a chapter for the journey. In the chapter list, click the option icon (top-left in the table) and tick the id.
        Get the id of the chapter. This is the chapter_id you need.

How to use this code:
    Option 1 using Pycharm:
        Just copy this file to your Pycharm project folder and you willl see lucybot.py file under your project.
        Change username, passwd, and chapter_id below.
        Shift-Ctr-F10 to start to process your chapter messages.
        To stop your script on Pycharm, click on the stop button on the right of the Run console.
        Monitor any errors
        Modify the code to make it more interesting

    Option 2 using any text editor:
        Change username, passwd, and chapter_id below.
        Suppose you have install Pyhton 2.7 in C:/python27
        You can run this code directly using command line:
            C:\python27\python lucybot.py
        To stop your script, press Ctrl-C

"""
#================================================
# Global variables that you need to change
username = "sun57759"  # change this to your username
passwd = "hx920309"    # change this to your passwd
chapter_id = 49           # change this to your chapter ID. It is INT type.
pathToDataFolder = "d:/dataFolder"  # change this to the folder where you want to place downloaded files for processing

#================================================
# import libraries
__author__ = 'Docentron PTY LTD'
import time
from lucy.message import Message

import pickle
from lucy import text

#================================================
# function definitions
msgClassifier = None
vocab = None
vocabidf = None
def makeReply(msg, fdata=None):
    """
    Compose a reply of the message.
    Use all information we have to provide the best reply for the msg that can help the student

    :param msg: the message for which we need to reply
    :param fdata: file data attached to the message if any
    :param classifier: classifier if any
    :param vocab: vocabulary if any
    :param vocabibf: idf of each terms in the vocab if any
    :return: the message content for the receiver.
    """
    global msgClassifier, vocab, vocabidf

    # do something amazing to generate the content for msg

    # now compose the content for replying
    mcontent = msg['content']
    fv = text.genfeatureVectorFromString(mcontent, vocab, vocabidf)
    pr = msgClassifier.predict([fv])        # prediction (0=closed question or 1=open question)
    pp = msgClassifier.predict_proba([fv])  # probability of each of the classes
    print pr
    print pp

    if pr[0] > 0:
        content = "Open question"
    else:
        content = "Closed question"
    content += ": pp={0}, msg={1}".format(pp[0][1], mcontent)
    return content

def buildDatabase(messages):
    # this is called when retrieved messages from the server
    # we can save the messages to a local database and analyse patterns of users
    pass

#================================================
# this is where the program starts
if __name__ == '__main__':
    #------------------------------------
    # Load a msg classification model and the vocabulary for processing text messages
    msgClassifier = pickle.load(open("demos/usenet_questions/models/data_questions_model.p", "rb"))
    r = text.readvocab("demos/usenet_questions/models/data_questions_vocab.txt")
    vocab = r[0]
    vocabtf = r[1]
    vocabidf = r[2]

    #------------------------------------
    # create LUCY API object with your username and password for a chapter (lesson) of a journey
    # You will need to create a journey and add a chapter to the journey at KOPO MES. Get the chapter ID as the lesson_id
    # The chapter should NOT contain any questions.
    lucy = Message(username = username, passwd = passwd, lesson_id = chapter_id, pathToDataFolder = pathToDataFolder)
    lucy.processMessages(makeReply, buildDatabase)

    # try these questions:
    #   Is this true?        closed question
    #   How can I fix this?  open question