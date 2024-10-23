# 5_5_sunspot.py
import numpy as np
import keras
import pandas as pd
import csv
import nltk
import pandas as pd
from sklearn import preprocessing, model_selection


# 텐서플 시험문제
# 월간 흑점 데이터에 대해 rnn모델을 만드세요
# 시퀀스 길이: 30개
# 학습 3000개
# 검사: 나머지
# 통과 기준: mae: 0.12

def make_xy(seq_length):
    df = pd.read_csv('data/monthly-sunspots.csv', engine='python',
                     delimiter=',', index_col=0)
    print(df)
    print(df.values[:, 0:].shape)
    sunspots = preprocessing.minmax_scale(df.values[:, 0:])
    grams = nltk.ngrams(sunspots, 31)
    grams = list(grams)
    print(np.array(list(grams)).shape)

    x = [i[:-1] for i in grams]
    y = [i[-1:]for i in grams]
    # print(x)
    print(np.array(x).shape)
    print(np.array(y).shape)
    print(type(x))
    print(len(x))
    # exit()
    return np.array(x), np.array(y)


def rnn_final(seq_length):
    x, y = make_xy(seq_length)

    # print(vocab)
    print(x.shape)
    print(y.shape)
    data = model_selection.train_test_split(x, y, train_size=2000)
    # exit()
    x_train, x_test, y_train, y_test = data
    model = keras.Sequential([
        keras.layers.Input(shape=x.shape[1:]),  # 데이터 1개의 shape, x[0].shape, shape[1:]
        # keras.layers.SimpleRNN(16, return_sequences=False),
        # keras.layers.SimpleRNN(32, return_sequences=False),
        keras.layers.GRU(32, return_sequences=False),
        # keras.layers.Dense(len(vocab), activation='softmax'),
        keras.layers.Dense(1),

    ])
    # model = keras.Sequential()
    # model.add(keras.layers.Dense(1))  # 곱셉이 있는 layer
    model.summary()

    # exit()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1),
                  loss=keras.losses.mean_squared_error,
                  # loss=keras.losses.categorical_crossentropy,
                  # loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['mae'])

    # 학습
    # model.fit(x, y, epochs=100, verbose=2)
    model.fit(x_train, y_train, epochs=30, verbose=2, validation_data=(x_test, y_test))

    # 사용
    p = model.predict(x_test, verbose=2)  # 예측
    print(p)
    # y_arg = np.argmax(y, axis=1)
    # p_arg = np.argmax(p, axis=2)  # 차원이 달라져서 axis를 바꿈
    # print(sentence)
    # print('_', ''.join(vocab[p_arg[0]]), end='', sep='')
    # for i in range(1, len(p_arg)):
    #     pp = p_arg[i]
    #     print(vocab[pp][-1], end='')
    print()


if __name__ == '__main__':
    rnn_final(seq_length=20)
