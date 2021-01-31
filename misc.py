from lab import*
import random

if __name__ == '__main__':
    dims = (10, 10)
    bombs = []
    for i in range(30):
        bombs += [tuple([random.randint(0, k - 1) for k in dims])]
    bombs = [(3, 3, 0, 1), (3, 4, 1, 2), (0, 2, 3, 1), (2, 6, 1, 0), (1, 4, 4, 0), (1, 3, 2, 1), (3, 2, 3, 3), (0, 2, 2, 0), (3, 1, 1, 1), (1, 3, 0, 1), (1, 0, 4, 0), (2, 3, 3, 2), (2, 0, 1, 1), (1, 4, 2, 1), (3, 6, 4, 0), (2, 2, 2, 0), (2, 1, 3, 3), (2, 0, 0, 0), (2, 4, 3, 0), (0, 4, 4, 3), (3, 0, 1, 3), (2, 3, 2, 2), (3, 3, 1, 2), (3, 2, 3, 1), (3, 6, 0, 0), (3, 4, 4, 1), (1, 2, 4, 0), (3, 3, 4, 1), (2, 6, 0, 1), (0, 6, 2, 1)]
    bombs = [(3, 3)]
    g = new_game_nd(dims, bombs)

    # (15, 15, 12, 10) 979
    # (4, 4, 2, 3, 3, 2) 373
    # (10, 10, 10, 10, 3) 988

    for i in render_nd(g):
        print(str(i))
    dig = (8, 8)
    print(get_element_nd(g['mask'], bombs[0]))
    print('REVEALED: ' + str(dig_nd(g, dig)))
    print(get_element_nd(g['mask'], bombs[0]))
    for i in render_nd(g):
        print(str(i))
