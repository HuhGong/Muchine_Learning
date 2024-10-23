# util.py
import numpy as np


def insert_text(lines, row, col, text):
    line_row = row * 3 + 1
    line_col = col * 12 + 4

    s = lines[line_row]
    s = lines[line_row] = s[:line_col] + text + s[line_col + len(text):]


def make_memory_table(q_table, size):
    q1 = np.zeros([len(q_table), 3, 3])  # (16, 3, 3)
    for v, t in zip(q1, q_table):  # v: (3, 3) t: (4,)
        v[1, 0] = t[0]
        v[2, 1] = t[1]
        v[1, 2] = t[2]
        v[0, 1] = t[3]

    q2 = q1.reshape(size, size, 3, 3)  # (4, 4, 3, 3)

    lines = []
    for i in range(4):
        for j in range(3):
            flat = np.reshape(q2[i, :, j, :], newshape=-1)
            concat = ['{:.2f}'.format(r) if r else ' -- ' for r in flat]
            # concat = ['{:.2f}'.format(r) for r in flat]
            lines.append(''.join(concat))
    return lines


def draw_value_table(lines):
    size = len(lines) // 3
    print('+' + '-' * (12 * size + 3 * (4 - 1) + 1) + '+')

    for i, line, in enumerate(lines):
        print('|', end='')
        for j, c in enumerate(line):
            print(c, end='')
            if j % 12 == 11:
                print('|', end='')
        print()
        if i % 3 == 2:
            print('+' + '-' * (12 * size + 3 * (4 - 1) + 1) + '+')


def draw_q_table(q_table):
    lines = make_memory_table(q_table, size=4)
    insert_text(lines, 0, 0, 'STRT')
    insert_text(lines, 0, 0, 'GOAL')

    for r, c in [(1, 1), (1, 3), (2, 3), (3, 0)]:
        insert_text(lines, r, c, 'HOLE')

    draw_value_table(lines)
