# 5_2_linerud.py
import numpy as np
from sklearn import datasets
from tensorflow.keras.layers import Input, Dense
import keras


class EpochStep(keras.callbacks.Callback):
    def __init__(self, step=500):
        super().__init__()
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch % self.step == 0:
            print(epoch, logs)
            # print('{:5} : acc {:.03f} loss {:.03f}'.format(epoch, logs['acc'], logs['loss']))


data = datasets.load_linnerud()
print(data['feature_names'])  # ['Chins', 'Situps', 'Jumps']
print(data['target_names'])  # ['Weight', 'Waist', 'Pulse']

# 퀴즈 sklearn에 있는 linnerrud 데이터 셋에 대해
# 함수형 모델을 구축하세요
x, y = datasets.load_linnerud(return_X_y=True)
# x = (data['data'])
# y = (data['target'])
print(x)
print(y)

# model = keras.Sequential([
#     # keras.layers.Dense(3, input_shape=(1, 1)),
#     Input(shape=x[0].shape),
#     Dense(16, 'relu'),
#     Dense(3),
# ])
inputs = keras.layers.Input(shape=x.shape[1:])
output = keras.layers.Dense(16, 'relu')(inputs)
output = keras.layers.Dense(3)(output)

output1 = keras.layers.Dense(4, activation='relu', name='hi1')(output)
output1 = keras.layers.Dense(1, activation='relu', name='hi2')(output1)
output2 = keras.layers.Dense(4, activation='relu', name='hi3')(output)
output2 = keras.layers.Dense(1, activation='relu', name='hi4')(output2)
output3 = keras.layers.Dense(4, activation='relu', name='hi5')(output)
output3 = keras.layers.Dense(1, activation='relu', name='hi6')(output3)

model = keras.Model(inputs, [output1, output2, output3])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss=keras.losses.mean_squared_error,
              metrics=['mae', 'mae', 'mae'])
model.summary()

model.fit(x, (y[:, 0:1], y[:, 1:2], y[:, 2:3]), epochs=200, verbose=0,
          validation_data=(x, y),
          callbacks=[EpochStep(100)]),
