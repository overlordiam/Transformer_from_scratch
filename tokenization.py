import torch
import numpy as np

class Tokenizer(object):
    def __init__(self, english_sentences, kannada_sentences, max_sentence_length, english_vocabulary, kannada_vocabulary):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences
        self.max_sentence_length = max_sentence_length
        self.english_vocabulary = english_vocabulary
        self.kannada_vocabulary = kannada_vocabulary

    def text_preprocess(self):
        indices_to_english = {k:v for k, v in enumerate(self.english_vocabulary)}
        english_to_indices = {v:k for k, v in enumerate(self.english_vocabulary)}
        kannada_to_indices = {k:v for k, v in enumerate(self.kannada_vocabulary)}
        indices_to_kannada = {k:v for k, v in enumerate(self.kannada_vocabulary)}

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
        # print(len(english_sentences), len(kannada_sentences))
        for index in range(len(self.english_sentences)):
            english_sentence, kannada_sentence = self.english_sentences[index], self.kannada_sentences[index]
            if self.is_valid(english_sentence, self.english_vocabulary, self.max_sentence_length) and \
                self.is_valid(kannada_sentence, self.kannada_vocabulary, self.max_sentence_length):
                # print(index)
                valid_indices.append(index)
        # print(valid_indices[:10])
            
        return valid_indices
    





if __name__ == '__main__':
    START_TOKEN = ''
    PADDING_TOKEN = ''
    END_TOKEN = ''

    kannada_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
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
                        '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', PADDING_TOKEN, END_TOKEN]

    english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            ':', '<', '=', '>', '?', '@', 
                            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                            'Y', 'Z',
                            '[', ']', '^', '_', '`',
                            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                            'y', 'z', 
                            '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]


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
    

                
    valid_sentence_indices = tokenizer.valid_sentences()
    print(len(valid_sentence_indices))