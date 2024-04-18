import json
import re
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, LSTM
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import pickle
import os 

from usefullFunctions import create_vocab, texts_to_sequences

class ModelTraining():
    
    def __init__(self, JSON_file=""):
        self.data = []
        self.filename = ""
        self.savePath = os.path.join(os.getcwd(), 'bin/')
        self.basicsIntents = []
        
        if JSON_file !="":
            path = os.path.join(os.getcwd(), JSON_file)
            self.verify_file(JSON_file)
            
        else :
            print("Erreur: aucun fichier intents.json spécifier")
        
            
    def verify_file(self, filename):
        if not os.path.exists(filename):
           print("Erreur: le fichier "+ self.filename + " n'existe pas")
           exit()
           
        else:
            with open(filename, 'r') as file:
                self.data = json.load(file)
                self.basicsIntents = [intent['tag'] for intent in self.data['intents']]
        self.filename = filename
    
    def reload(self):
        with open(self.filename, 'r') as file:
            self.data = json.load(file)
            self.basicsIntents = [intent['tag'] for intent in self.data['intents']]
        
    def train(self, vocab_size=3000, embedding_dim=32, max_len=40, oov_token="<OOV>", epochs=500):
        
        training_sentences = []
        training_labels = []
        labels = []
        responses = []

        

        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                clean_pattern = re.sub(r'[^a-zA-Z0-9\s]', '', pattern.lower())
                training_sentences.append(clean_pattern)
                training_labels.append(intent['tag'].lower())
            clean_responses = [re.sub(r'[^a-zA-Z0-9\s]', '', response.lower()) for response in intent['responses']]
            responses.append(clean_responses)

            
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

        num_classes = len(labels)
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(training_labels)
        training_labels = lbl_encoder.transform(training_labels)

        vocab = create_vocab(training_sentences, vocab_size, oov_token)
        sequences = texts_to_sequences(training_sentences, vocab, oov_token)
        padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Dropout(0.5),
            LSTM(64, return_sequences=True),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(lbl_encoder.classes_), activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
        model_path = os.path.join(self.savePath, 'chat_model.keras')
        model.save(model_path)

        vocab_path = os.path.join(self.savePath, 'vocab.pickle')
        lbl_path = os.path.join(self.savePath, 'label_encoder.pickle')
        
        with open(vocab_path, 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(lbl_path, 'wb') as ecn_file:
            pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        return model_path, vocab_path, lbl_path, self.basicsIntents
