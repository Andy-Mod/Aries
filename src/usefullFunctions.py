from collections import Counter
import json 
import re
import unicodedata
import string


def retirer_accents_et_autres(chaine):
    table_ponctuation = str.maketrans('', '', string.punctuation)
    chaine_normalisee = unicodedata.normalize('NFD', chaine)
    
    chaine_sans_accents = ''.join([c for c in chaine_normalisee if unicodedata.category(c) != 'Mn'])
    texte_sans_ponctuation = chaine_sans_accents.translate(table_ponctuation)
    
    return texte_sans_ponctuation.lower().strip()

def create_vocab(texts, num_words, oov_token):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    most_common = counter.most_common(num_words - 1)
    vocab = {word: idx + 1 for idx, (word, count) in enumerate(most_common)}
    vocab[oov_token] = 0
    return vocab

def texts_to_sequences(texts, vocab, oov_token):
    if vocab is None:
        raise ValueError("The vocab parameter is None. Ensure the vocab is properly initialized.")
    
    sequences = []
    for text in texts:
        sequence = []
        for word in text.lower().split():
            sequence.append(vocab.get(word, vocab.get(oov_token, 1)))  # Default to 1 if oov_token is not found
        sequences.append(sequence)
    return sequences

def read_file_to_array(filepath):
    lines_array = []
    with open(filepath, 'r') as file:
        for line in file:
            lines_array.append(line.strip())
    return lines_array

def loadBasics(jsonPath):
    data = []
    basicsIntents = []
    with open(jsonPath, 'r') as file:
        data = json.load(file)
        basicsIntents = [intent['tag'] for intent in data['intents']]
    return basicsIntents, data

def extract_vocab_size(json_file):
    """
    Extrait la taille du vocabulaire à partir d'un fichier JSON contenant des intents.
    
    :param json_file: Chemin vers le fichier JSON.
    :return: Taille du vocabulaire (nombre de mots uniques).
    """
    vocab = Counter()
    
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        if "intents" in data:
            for intent in data["intents"]:
                for text in intent.get("patterns", []) + intent.get("responses", []):
                    words = re.findall(r'\w+', text.lower())  # Tokenisation basique
                    vocab.update(words)
    
    return len(vocab)