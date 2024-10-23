# 5_6_sarcasm.py
import json
import keras
import numpy as np
from sklearn import model_selection
import tensorflow.keras as tf_keras


# 텐서플로 시험 문제
# sarcasm.json 파일에 대해
# 2만개로 학습하고 나머지에 대해 정확도를 구하세요
# 통과 기준: 80%

# 단어장: 2000개
# 시퀀스 길이: 200

# "article_link": 사용 안함
# "headline": 문자열 제목
# "is_sarcastic": Boolean
with open('data/Sarcasm.json', 'r') as file:

    headlines, targets = [], []
    for line in file:
        d = json.loads(line.strip())
        # print(type(d))

        headlines.append(d['headline'])
        targets.append(int(d['is_sarcastic']))

    file.close()

print(headlines[:3])
print(targets[:3])

targets = np.reshape(targets, newshape=(-1, 1))

# --------------------------------------- #

vocab_size, seq_len = 2000, 200

tokenizer = tf_keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(headlines)

seq = tokenizer.texts_to_sequences(headlines)
pad = keras.preprocessing.sequence.pad_sequences(seq, maxlen=seq_len)

data = model_selection.train_test_split(pad, targets, train_size=20000)
x_train, x_test, y_train, y_test = data

model = keras.Sequential([
    keras.layers.Input(shape=x_train.shape[1:]),
    keras.layers.Embedding(vocab_size, 100),
    # keras.layers.LSTM(50, return_sequences=False),
    keras.layers.Bidirectional(keras.layers.LSTM(50, return_sequences=False)),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=1, verbose=1, batch_size=128,
          validation_data=(x_test, y_test))




