from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle 
from streamlit import *  # type: ignore
import numpy as np

model = load_model('models/best_model_acc.keras')
tokenizer = pickle.load(open('pipelines/tokenizer.pkl', 'rb'))

def generate_text(model, tokenizer, seed_text, max_len, predict_next_words=2, temp=1.0, topK=0, topP=1.0):
    def respond(preds, temp=1.0, topK=0, topP=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8) / temp
        exp_preds = np.exp(preds - np.max(preds))
        preds = exp_preds / np.sum(exp_preds)
        if topK > 0:
            indices = np.argpartition(preds, -topK)[-topK:]
            probs = np.zeros_like(preds)
            probs[indices] = preds[indices]
            preds = probs / np.sum(probs) if np.sum(probs) > 0 else preds
        if topP < 1.0:
            sorted_indices = np.argsort(preds)[::-1]
            sorted_probs = np.sort(preds)[::-1]
            cum_probs = np.cumsum(sorted_probs)
            cutoff = cum_probs <= topP
            cutoff_indices = sorted_indices[cutoff]
            probs = np.zeros_like(preds)
            probs[cutoff_indices] = preds[cutoff_indices]
            preds = probs / np.sum(probs) if np.sum(probs) > 0 else preds
        preds = np.nan_to_num(preds)
        preds = preds / np.sum(preds) if np.sum(preds) > 0 else np.ones_like(preds)/len(preds)
        return np.random.choice(len(preds), p=preds)

    input_text = seed_text
    for _ in range(predict_next_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding="pre")
        preds = model.predict(token_list, verbose=0)[0]
        next_index = respond(preds, temp=temp, topK=topK, topP=topP)
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                input_text += " " + word
                break
    return input_text

set_page_config('RNN Based Chatbot', layout='centered', initial_sidebar_state='expanded', page_icon='🤖')
title('RNN Based Chatbot with Adjustable Metrics')

with sidebar:   # type: ignore
    info('This is a small scale RNN model based on the Shakespeare-Hamlet.\nYou can adjust the temperature, top K and top P of the model (Basically Fine Tuning)')
    temp = float(slider(label='Temperature (0.0 - 2.0)', min_value=0.0, max_value=2.0, value=1.0, step=0.01))
    topK_input = text_input(label='Top K', value="5")
    topK = int(topK_input) if topK_input.isdigit() and 1 <= int(topK_input) <= 20 else 5
    max_output_input = text_input(label='Generated Words', value="1")
    words = int(max_output_input) if max_output_input.isdigit() else 1
    words = max(1, min(words, 50))
    topP = float(slider(label='Top P (0.0 - 1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01))

if 'history' not in session_state:
    session_state.history = []

for message in session_state.history:
    with chat_message(message['role']):
        markdown(message['content'])

if user_input := chat_input(placeholder='Type Here'):
    session_state.history.append({'role':'user',"content":user_input})
    with chat_message('user'):
        markdown(user_input)
    with chat_message('assistant'):
        with spinner('Thinking', show_time=True):
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                seed_text="Barnardo.",
                max_len=model.input_shape[1] + 1,
                predict_next_words=words,
                temp=temp,
                topK=topK,
                topP=topP
            )
            markdown(generated_text)
        session_state.history.append({'role':'assistant','content':generated_text})
else:
    info('Type some text from the play (Hamlet - Shakespeare)')
