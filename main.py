import gradio as gr
import time
import json
import numpy as np
import nltk
import random
import tensorflow as tf

from sentence_transformers import SentenceTransformer, util
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer

MODEL_TF = './model/model.h5'
MODEL_TORCH = './model/model_transformer/'


def load_intent_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


intent_data = load_intent_data('./dataset/Datasets_copy.json')
stemmer = StemmerFactory().create_stemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = SentenceTransformer(MODEL_TORCH)
modelTF = tf.keras.models.load_model(MODEL_TF)
nltk.download('wordnet', download_dir='./model/nltk')


def create_document(intents):
    all_words = []
    all_tags = []
    documents = []
    ignore = ['!', '.', '?', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = tokenizer.encode(pattern, add_special_tokens=True)
            w = [tokenizer.decode(w_token) for w_token in w if tokenizer.decode(
                w_token) not in ignore]

            w = [stemmer.stem(word) for word in w]

            all_words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in all_tags:
                all_tags.append(intent['tag'])

    all_words = [w.lower() for w in all_words if w not in ignore]
    all_words = sorted(list(set(all_words)))
    all_tags = sorted(list(set(all_tags)))
    return all_words, all_tags, documents


total_words, total_tags, total_documents = create_document(intent_data)


def clean_up_userQuery(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    sentence_words = [wordnet_lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def create_BoW(sentence, words, show_details=False):
    sentence_words = clean_up_userQuery(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details == True:
                    print("Word Found in the Bag : %s" % w)
    return (np.array(bag))


def match_intent(input_token, intent_data):
    input_embeddings = model.encode(input_token, convert_to_tensor=True)

    best_match = None
    best_similarity = -1

    for intent in intent_data['intents']:
        for pattern in intent['patterns']:
            pattern_embedding = model.encode(pattern, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(
                input_embeddings, pattern_embedding)[0].item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (intent, pattern, similarity)

    return best_match


context = {}
ERROR_TRESHOLD = 0.50
words = total_words
classes = total_tags


# def chat():
# print("(Mulai Percakapan dengan Bot(quit/q/bye) untuk stop !)")
# while True:
#     inp = input("Kamu : ")
#     inp = inp.lower()
#     if inp == "quit" or inp == "q" or inp == "bye":
#         break
#     else:
#         best_match = match_intent(inp, intent_data)
#         if best_match is not None:
#             matched_intent, matched_pattern, similarity = best_match
#             print(
#                 f"Token input cocok dengan intent: {matched_intent['tag']}")
#             print(f"Pola terbaik: {matched_pattern}")
#             print(f"Kemiripan: {similarity * 100:.2f}%")
#             if similarity * 100 > 80:
#                 send_matched_patterns(matched_pattern)
#         else:
#             print("Bisa jelaskan lebih detail lagi ??")

def send_matched_patterns(inp):
    inp = inp.lower()
    res = response(inp, show_details=False)
    cls = classify(inp)
    return res, cls


def classify(sentence):
    p = create_BoW(sentence, words, show_details=False)
    results = modelTF.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_TRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def response(sentence, userID='NCI', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intent_data['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details:
                            print('context:', i['context_set'])
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details:
                            print('Tag:', i['tag'])
                        return random.choice(i['responses'])
            results.pop(0)


def chatbot(inp, history):
    for i in range(len(inp)):
        time.sleep(0.1)
        inp = inp.lower()
        if inp == "quit" or inp == "q" or inp == "bye":
            yield "Terima kasih sudah menggunakan chatbot ini"
        else:
            best_match = match_intent(inp, intent_data)
            if best_match is not None:
                matched_intent, matched_pattern, similarity = best_match
                if similarity * 100 > 80:
                    send_matched_patterns(matched_pattern)
                    yield "bot menjawab: " + send_matched_patterns(matched_pattern)[0]


demo = gr.ChatInterface(chatbot, title="Chatbot").queue()

if __name__ == '__main__':
    demo.launch()
