import json
import colorama 
pip install colorama
colorama.init()
from colorama import Fore, Style, Back

# Define the JSON data
data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hey", "Is anyone there?", "Hello", "Hay"],
            "responses": ["Hello", "Hi", "Hi there"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["See you later", "Have a nice day", "Bye! Come back again"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful", "Thanks for the help"],
            "responses": ["Happy to help!", "Any time!", "My pleasure", "You're most welcome!"]
        },
        {
            "tag": "about",
            "patterns": ["Who are you?", "What are you?", "Who you are?"],
            "responses": ["I'm Lila, your bot assistant", "I'm Lila, an Artificial Intelligent bot"]
        },
        {
            "tag": "name",
            "patterns": ["what is your name", "what should I call you", "whats your name?"],
            "responses": ["You can call me Lila.", "I'm Lila!", "Just call me Lila"]
        },
        {
            "tag": "help",
            "patterns": ["Could you help me?", "give me a hand please", "Can you help?", "What can you do for me?", "I need support", "I need help", "support me please"],
            "responses": ["Tell me how can I assist you", "Tell me your problem to assist you", "Yes Sure, How can I support you"]
        },
        {
            "tag": "createaccount",
            "patterns": ["I need to create a new account", "how to open a new account", "I want to create an account", "can you create an account for me", "how to open a new account"],
            "responses": ["You can easily create a new account from our website", "Just go to our website and follow the guidelines to create a new account"]
        },
        {
            "tag": "complaint",
            "patterns": ["have a complaint", "I want to raise a complaint", "there is a complaint about a service"],
            "responses": ["Please provide your complaint in order to assist you", "Please mention your complaint, we will reach you and sorry for any inconvenience caused"]
        }
    ]
}

# Create and write the JSON file
with open('intents.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)


import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

with open('intents.json') as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
# to save the trained model
model.save("chat_model.keras")  


import pickle

# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model('chat_model.keras')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
