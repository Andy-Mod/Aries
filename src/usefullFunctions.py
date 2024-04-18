from collections import Counter
import json 

def create_vocab(texts, num_words, oov_token):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        most_common = counter.most_common(num_words - 1)
        vocab = {word: idx + 1 for idx, (word, count) in enumerate(most_common)}
        vocab[oov_token] = 0
        return vocab

def texts_to_sequences(texts, vocab, oov_token):
        sequences = []
        for text in texts:
            sequence = []
            for word in text.lower().split():
                sequence.append(vocab.get(word, vocab[oov_token]))
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