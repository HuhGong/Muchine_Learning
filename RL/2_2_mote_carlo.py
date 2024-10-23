# 2_2_mote_carlo.py
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)  # 소수점 자리수 조절

# 퀴즈
# 몬테카를로 방법론을 사용해서 원주율을 계산하세요

N, cnt = 100000, 0
# N, cnt = 10<<7, 0
inner, outter = [], []

for _ in range(N):
    x = np.random.rand(1)
    y = np.random.rand(1)
    cnt += (x ** 2 + y ** 2 < 1)

    # print(cnt)
    if x ** 2 + y ** 2 < 1:
        inner.append((x, y))
    else:
        outter.append((x, y))


print(cnt * 4 / N)

# # 느린 코드

# for x, y in inner:
#     plt.plot(x, y, 'bo')
# for x, y in outter:
#     plt.plot(x, y, 'ro')

#  빠른 코드
plt.plot(*list(zip(*inner)))
plt.plot(*list(zip(*outter)))
plt.show()


# a = [(1, -2), (3, -5), (4, -9)]
# b = list(zip(*a))
# print(b)
