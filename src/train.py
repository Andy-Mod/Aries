import json
import re
import numpy as np
import tensorflow as tf
import pickle
import os
from keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Embedding, Dropout, LSTM, Bidirectional, Conv1D, 
    MaxPooling1D, GlobalMaxPooling1D, BatchNormalization
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from usefullFunctions import create_vocab, texts_to_sequences


class ModelTraining:
    def __init__(self, JSON_file=""):
        self.data = []
        self.filename = ""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.main_dir = os.path.join(self.dir, '../')
        
        self.model_path = os.path.join(self.main_dir, "bin/chat_model.keras")
        self.vocab_path = os.path.join(self.main_dir, "bin/vocab.pickle")
        self.lbl_path = os.path.join(self.main_dir, "bin/label_encoder.pickle")

        self.savePath = os.path.join(self.main_dir, 'bin/')
        os.makedirs(self.savePath, exist_ok=True)

        self.basicsIntents = []
        
        if JSON_file:
            self.filename = JSON_file
            self.verify_file(self.filename)
        else:
            raise FileNotFoundError("Erreur: aucun fichier intents.json spécifié")
            
    def verify_file(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Erreur: le fichier {filename} n'existe pas")
        
        with open(filename, 'r') as file:
            self.data = json.load(file)
            self.basicsIntents = [intent['tag'] for intent in self.data['intents']]
        self.filename = filename
    
    def reload(self):
        with open(self.filename, 'r') as file:
            self.data = json.load(file)
            self.basicsIntents = [intent['tag'] for intent in self.data['intents']]
    
    def train(self, vocab_size=3000, embedding_dim=32, max_len=40, oov_token="<OOV>", epochs=50, patience=5):
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

        # Ajout explicite de la classe "unknown"
        if "unknown" not in labels:
            labels.append("unknown")
        
        num_classes = len(labels)
        
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(labels)
        training_labels = lbl_encoder.transform(training_labels)
        training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=num_classes)

        vocab = create_vocab(training_sentences, vocab_size, oov_token)
        sequences = texts_to_sequences(training_sentences, vocab, oov_token)
        padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Conv1D(512, 5, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))),
            Dropout(0.4),
            Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.005))),
            Dropout(0.4),
            Dense(128, activation='relu', kernel_regularizer=l2(0.005)),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            metrics=['accuracy']
        )
        model.summary()

        # early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        model.fit(
            padded_sequences, np.array(training_labels),
            validation_split=0.2,
            epochs=epochs,
            batch_size=32,
            # callbacks=[early_stopping]
        )

        model.save(self.model_path)
        
        with open(self.vocab_path, 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(self.lbl_path, 'wb') as ecn_file:
            pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.model_path, self.vocab_path, self.lbl_path, self.basicsIntents
