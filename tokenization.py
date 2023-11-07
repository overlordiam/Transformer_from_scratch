import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Tokenizer(object):
    def __init__(self, english_sentences, kannada_sentences, max_sentence_length, english_vocabulary, kannada_vocabulary):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences
        self.max_sentence_length = max_sentence_length
        self.english_vocabulary = english_vocabulary
        self.kannada_vocabulary = kannada_vocabulary

    def text_preprocess(self):
        self.indices_to_english = {k:v for k, v in enumerate(self.english_vocabulary)}
        self.english_to_indices = {v:k for k, v in enumerate(self.english_vocabulary)}
        self.kannada_to_indices = {k:v for k, v in enumerate(self.kannada_vocabulary)}
        self.indices_to_kannada = {k:v for k, v in enumerate(self.kannada_vocabulary)}

        self.english_sentences = [sentence.rstrip() for sentence in self.english_sentences]
        self.kannada_sentences = [sentence.rstrip() for sentence in self.kannada_sentences]


    def is_valid(self, sentence, vocabulary, max_sentence_length):
        for token in list(set(sentence)):
            if token not in vocabulary:
                return False
        if len(list(sentence)) > (max_sentence_length - 1):
                return False
        return True

    def valid_sentences(self):
        self.text_preprocess()
        valid_indices = []
        for index in range(len(self.english_sentences)):
            english_sentence, kannada_sentence = self.english_sentences[index], self.kannada_sentences[index]
            if self.is_valid(english_sentence, self.english_vocabulary, self.max_sentence_length) and \
                self.is_valid(kannada_sentence, self.kannada_vocabulary, self.max_sentence_length):
                valid_indices.append(index)

        self.english_sentences = [self.english_sentences[i] for i in valid_indices]
        self.kannada_sentences = [self.kannada_sentences[i] for i in valid_indices]
    
        # print(len(self.english_sentences), len(self.kannada_sentences))

    def tokenize(self, sentence, start_token=True, end_token=True):
        tokenized_sentence = [self.english_to_indices[token] for token in list(sentence)]
        if start_token:
            tokenized_sentence.insert(0, self.english_to_indices['START_TOKEN'])
        if end_token:
            tokenized_sentence.append(self.english_to_indices['END_TOKEN'])
        for _ in range(len(sentence), self.max_sentence_length):
            tokenized_sentence.append(self.english_to_indices['PADDING_TOKEN'])
        return torch.tensor(tokenized_sentence)
        

class TextDataset(Dataset):
    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self,index):
        return (self.english_sentences[index], self.kannada_sentences[index])


if __name__ == '__main__':
    START_TOKEN = ''
    PADDING_TOKEN = ''
    END_TOKEN = ''

    kannada_vocabulary = ['START_TOKEN', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ', 
                        'ँ', 'ఆ', 'ఇ', 'ా', 'ి', 'ీ', 'ు', 'ూ', 
                        'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ', 'ಌ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 
                        'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 
                        'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ', 
                        'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 
                        'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ', 
                        'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 
                        'ಯ', 'ರ', 'ಱ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ', 
                        '಼', 'ಽ', 'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ೕ', 'ೖ', 'ೞ', 'ೣ', 'ಂ', 'ಃ', 
                        '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', 'PADDING_TOKEN', 'END_TOKEN']

    english_vocabulary = ['START_TOKEN', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            ':', '<', '=', '>', '?', '@', 
                            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                            'Y', 'Z',
                            '[', ']', '^', '_', '`',
                            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                            'y', 'z', 
                            '{', '|', '}', '~', 'PADDING_TOKEN', 'END_TOKEN']



    # print(english_to_indices['b'])
    # print(indices_to_english[10])

    # print("ಕ" + "ೆ")

    with open('en-kn/train.en', 'r') as text:
        english_sentences = text.readlines()

    # print(english_sentences[:5])

    text = open('en-kn/train.kn', 'r', encoding='utf-8')
    kannada_sentences = text.readlines()
    text.close()

    # print(kannada_sentences[:5])

    # Ready the training set
    data_length = 1000

    english_sentences = english_sentences[:data_length]
    kannada_sentences = kannada_sentences[:data_length]

    # print(kannada_sentences[:5])
    tokenizer = Tokenizer(english_sentences=english_sentences, kannada_sentences=kannada_sentences, max_sentence_length=200,\
                           english_vocabulary=english_vocabulary, kannada_vocabulary=kannada_vocabulary)
    

                
    tokenizer.valid_sentences()
    # print(english_sentences[0])
    print(tokenizer.tokenize(tokenizer.english_sentences[0]))
