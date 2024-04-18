
import os
import re
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sr  

from train import ModelTraining
from usefullFunctions import texts_to_sequences, loadBasics
import pickle

class Chat:
    
    def __init__(self, jsonPath, train=False, max_len=10):
        
        if train:
            self.Model = ModelTraining(jsonPath)
            model_path, vocab_path, lbl_path, basics = self.Model.train()
            self.data = self.Model.data
            
        else:    
            model_path = os.path.join(os.getcwd(), "bin/chat_model.keras")
            vocab_path= os.path.join(os.getcwd(), "bin/vocab.pickle")
            lbl_path = os.path.join(os.getcwd(), "bin/label_encoder.pickle")
            
            self.model = keras.models.load_model(model_path)
            self.basicsIntents, self.data = loadBasics(jsonPath)
        
            
        self.max_len = max_len

        with open(vocab_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open(lbl_path, 'rb') as enc:
            self.lbl_encoder = pickle.load(enc)
            
    
    def predict_class(self, input, oov_token="<OOV>"):
        
        input = re.sub(r'[^a-zA-Z0-9\s]', '', input.lower())
        result = self.model.predict(keras.preprocessing.sequence.pad_sequences(texts_to_sequences([input], self.tokenizer, oov_token),
                                             truncating='post', maxlen=self.max_len))
        intent = self.lbl_encoder.inverse_transform([np.argmax(result)])
    
        print(intent)
        
        return intent    

    def answer(self, input, tag):
        ans = ""
        
        if tag in self.basicsIntents: 
            for i in self.data['intents']:
                if i['tag'] == tag:
                    answers = i['responses']
                    print(answers)
                    ans = np.random.choice(i['responses'])
        
        if tag == "":
            nonePath = os.path.join(os.getcwd(), 'data/none.txt')
            with open(nonePath, 'a') as none:
                none.write(input + '\n')
        
        print(ans)
        return ans
                
