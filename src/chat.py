import os
import re
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

from train import ModelTraining
from usefullFunctions import texts_to_sequences, loadBasics, extract_vocab_size

class Chat:
    def __init__(self, jsonPath, train=False, max_len=40, threshold=0.7):
        self.max_len = max_len
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self.lbl_encoder = None
        self.basicsIntents = []
        self.data = {}

        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.main_dir = os.path.join(self.dir, '../')
        
        self.model_path = os.path.join(self.main_dir, 'bin', 'chat_model.keras')
        self.vocab_path = os.path.join(self.main_dir, 'bin/vocab.pickle')
        self.lbl_path = os.path.join(self.main_dir, 'bin/label_encoder.pickle')

        if train:
            self.Model = ModelTraining(jsonPath)
            self.model_path, self.vocab_path, self.lbl_path, self.basicsIntents = self.Model.train(
                vocab_size=extract_vocab_size(jsonPath), epochs=190)
            self.data = self.Model.data

        if not all(os.path.exists(path) for path in [self.model_path, self.vocab_path, self.lbl_path]):
            raise FileNotFoundError("Erreur: Fichiers du modèle non trouvés. Entraînez le modèle avant de l'utiliser.")
        
        print("Chargement du modèle d'interprétation...")

        self.model = load_model(self.model_path)
        self.basicsIntents, self.data = loadBasics(jsonPath)

        with open(self.vocab_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        with open(self.lbl_path, 'rb') as enc:
            self.lbl_encoder = pickle.load(enc)

    def predict_class(self, input_text, oov_token="<OOV>"):
        print(f"Texte reçu pour interprétation : {input_text}")

        input_text = re.sub(r'[^a-zA-Z0-9\s]', '', input_text.lower())
        
        sequences = texts_to_sequences([input_text], self.tokenizer, oov_token)
        padded_sequences = keras.preprocessing.sequence.pad_sequences(
            sequences, truncating='post', maxlen=self.max_len)
        result = self.model.predict(padded_sequences)
        max_prob = np.max(result)

        tag = self.lbl_encoder.inverse_transform([np.argmax(result)])[0]

        print(f"Texte : {input_text}, Intention trouvée : {tag}, Confiance : {max_prob}")

        return tag if max_prob >= self.threshold else "unknown"

    def answer(self, input_text, tag):
        if tag == "unknown":
            none_path = os.path.join(self.main_dir, 'data/none.txt')
            os.makedirs(os.path.dirname(none_path), exist_ok=True)
            with open(none_path, 'a', encoding='utf-8') as file:
                file.write(input_text + "\n")
            return "Je ne comprends pas votre requête."

        for intent in self.data['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])

        return "Je ne trouve pas de réponse appropriée."
