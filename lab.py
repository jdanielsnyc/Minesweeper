#!/usr/bin/env python3
"""6.009 Lab -- Six Double-Oh Mines"""


# NO IMPORTS ALLOWED!

def dump(game):
    """
    Prints a human-readable version of a game (provided as a dictionary)
    """
    for key, val in sorted(game.items()):
        if isinstance(val, list) and val and isinstance(val[0], list):
            print(f'{key}:')
            for inner in val:
                print(f'    {inner}')
        else:
            print(f'{key}:', val)


# 2-D IMPLEMENTATION


def new_game_2d(num_rows, num_cols, bombs):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'mask' fields adequately initialized.

    Parameters:
       num_rows (int): Number of rows
       num_cols (int): Number of columns
       bombs (list): List of bombs, given in (row, column) pairs, which are
                     tuples

    Returns:
       A game state dictionary

    >>> dump(new_game_2d(2, 4, [(0, 0), (1, 0), (1, 1)]))
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: (2, 4)
    mask:
        [False, False, False, False]
        [False, False, False, False]
    state: ongoing
    """
    return new_game_nd((num_rows, num_cols), bombs)


def dig_2d(game, row, col):
    """
    Reveal the cell at (row, col), and, in some cases, recursively reveal its
    neighboring squares.

    Update game['mask'] to reveal (row, col).  Then, if (row, col) has no
    adjacent bombs (including diagonally), then recursively reveal (dig up) its
    eight neighbors.  Return an integer indicating how many new squares were
    revealed in total, including neighbors, and neighbors of neighbors, and so
    on.

    The state of the game should be changed to 'defeat' when at least one bomb
    is visible on the board after digging (i.e. game['mask'][bomb_location] ==
    True), 'victory' when all safe squares (squares that do not contain a bomb)
    and no bombs are visible, and 'ongoing' otherwise.

    Parameters:
       game (dict): Game state
       row (int): Where to start digging (row)
       col (int): Where to start digging (col)

    Returns:
       int: the number of new squares revealed

    >>> game = {'dimensions': (2, 4),
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'mask': [[False, True, False, False],
    ...                  [False, False, False, False]],
    ...         'state': 'ongoing'}
    >>> dig_2d(game, 0, 3)
    4
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: (2, 4)
    mask:
        [False, True, True, True]
        [False, False, True, True]
    state: victory

    >>> game = {'dimensions': [2, 4],
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'mask': [[False, True, False, False],
    ...                  [False, False, False, False]],
    ...         'state': 'ongoing'}
    >>> dig_2d(game, 0, 0)
    1
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: [2, 4]
    mask:
        [True, True, False, False]
        [False, False, False, False]
    state: defeat
    """

    """
    
    # OLD:
    
    if game['state'] == 'defeat' or game['state'] == 'victory':
        return 0

    game['mask'][row][col] = True

    if game['board'][row][col] == '.':
        # If we hit a bomb, we lose. Massive bruh moment.
        game['state'] = 'defeat'
        return 1

    revealed = 1
    h, w = game['dimensions']
    if game['board'][row][col] == 0:
        # If our selected square borders no bombs, we reveal the surrounding squares.
        for delta_r in [-1, 0, 1]:
            for delta_c in [-1, 0, 1]:
                r = row + delta_r
                c = col + delta_c
                if 0 <= r < h and 0 <= c < w:
                    if not game['mask'][r][c]:
                        # If square (r, c) has not already been revealed, we reveal it.
                        revealed += dig_2d(game, r, c)

    victory = True
    # Loop through every square on the board to determine if we've reached a victory state
    for r in range(h):
        for c in range(w):
            val = game['board'][r][c]
            visible = game['mask'][r][c]
            if not val == '.' and not visible:
                # If a non-bomb square is still hidden, we haven't won yet.
                victory = False
                break
    if victory:
        game['state'] = 'victory'

    return revealed
    """
    return dig_nd(game, (row, col))


def render_2d(game, xray=False):
    """
    Prepare a game for display.

    Returns a two-dimensional array (list of lists) of '_' (hidden squares), '.'
    (bombs), ' ' (empty squares), or '1', '2', etc. (squares neighboring bombs).
    game['mask'] indicates which squares should be visible.  If xray is True (the
    default is False), game['mask'] is ignored and all cells are shown.

    Parameters:
       game (dict): Game state
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    game['mask']

    Returns:
       A 2D array (list of lists)

    >>> render_2d({'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'mask':  [[False, True, True, False],
    ...                   [False, False, True, False]]}, False)
    [['_', '3', '1', '_'], ['_', '_', '1', '_']]

    >>> render_2d({'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'mask':  [[False, True, False, True],
    ...                   [False, False, False, True]]}, True)
    [['.', '3', '1', ' '], ['.', '.', '1', ' ']]
    """

    """
    h, w = game['dimensions']
    board = game['board']
    board_render = [[0] * w for i in range(h)]
    for r in range(h):
        for c in range(w):
            val = board[r][c]
            board_render[r][c] = (' ' if val == 0 else str(val)) if game['mask'][r][c] or xray else '_'
    return board_render
    """
    return render_nd(game, xray)


def render_ascii(game, xray=False):
    """
    Render a game as ASCII art.

    Returns a string-based representation of argument 'game'.  Each tile of the
    game board should be rendered as in the function 'render_2d(game)'.

    Parameters:
       game (dict): Game state
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    game['mask']

    Returns:
       A string-based representation of game

    >>> print(render_ascii({'dimensions': (2, 4),
    ...                     'state': 'ongoing',
    ...                     'board': [['.', 3, 1, 0],
    ...                               ['.', '.', 1, 0]],
    ...                     'mask':  [[True, True, True, False],
    ...                               [False, False, True, False]]}))
    .31_
    __1_
    """
    board_string = ''
    h, w = game['dimensions']
    for r in range(h):
        for c in range(w):
            val = game['board'][r][c]
            board_string += (' ' if val == 0 else str(val)) if game['mask'][r][c] or xray else '_'
        if r < h - 1:
            board_string += '\n'
    return board_string


# N-D IMPLEMENTATION


def new_game_nd(dimensions, bombs):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'mask' fields adequately initialized.


    Args:
       dimensions (tuple): Dimensions of the board
       bombs (list): Bomb locations as a list of lists, each an
                     N-dimensional coordinate

    Returns:
       A game state dictionary

    >>> g = new_game_nd((2, 4, 2), [(0, 0, 1), (1, 0, 0), (1, 1, 1)])
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    mask:
        [[False, False], [False, False], [False, False], [False, False]]
        [[False, False], [False, False], [False, False], [False, False]]
    state: ongoing
    """

    board = build_nd(dimensions, 0)  # Initialize board nested list
    lower = tuple([0] * len(dimensions))  # The lower bound for any coordinate in our n-dimensional minefield
    upper = tuple([i - 1 for i in dimensions])  # The upper bound for any coordinate in our n-dimensional minefield
    for bomb_coord in bombs:
        set_element_nd(board, bomb_coord, '.')  # Place a bomb at it's designated position
        # Iterate through each coordinate bordering the placed bomb, increasing the number of bombs
        # that square borders by one
        neighbors = get_neighbors(bomb_coord)
        for border_coord in get_neighbors(bomb_coord):
            if bounded_nd(border_coord, lower, upper) and not get_element_nd(board, border_coord) == '.':
                # If this coordinate is within the grid and isn't a bomb, we update the number of
                # bombs it borders
                count = get_element_nd(board, border_coord) + 1
                set_element_nd(board, border_coord, count)

    mask = build_nd(dimensions, False)  # Initialize mask nested list
    return {
        'dimensions': dimensions,
        'board': board,
        'mask': mask,
        'state': 'ongoing',
    }


def build_nd(dims, val=0):
    """
    Recursively builds an n-dimensional nested list representing a tensor

    Parameters:
        dims: an n-dimensional tuple representing the dimensions a tensor
        val: the default value that every element of the tensor created is initialized to
    """

    if len(dims) == 0:
        # A zero-dimensional tensor is a scalar
        return val

    tensor = [val] * dims[0]
    for i in range(len(tensor)):
        # Create the next dimension of our tensor
        tensor[i] = build_nd(dims[1:], val)
    return tensor


def get_element_nd(nd_list, coord):
    """
    Returns the element of an n-dimensional nested list at a specified position

    Parameters:
        nd_list: an n-dimensional nested list representing a tensor of elements
        coord: a tuple representing the coordinate of a specific element in nd_list
    """

    element = nd_list
    for c in coord:
        element = element[c]
    return element


def set_element_nd(nd_list, coord, elt):
    """
    Sets the element of an n-dimensional nested list at a specified position

    Parameters:
        nd_list: an n-dimensional nested list representing a tensor of elements
        coord: a tuple of length n representing the coordinate of the element to be inserted
        elt: the element being inserted into nd_list
    """

    if len(coord) == 1:
        # If our list of coordinates is only one value long, it means we've reached
        # the dimension of the element we want to set
        nd_list[coord[0]] = elt
    else:
        set_element_nd(nd_list[coord[0]], coord[1:], elt)


def get_neighbors(center):
    """
    Returns a tuple of coordinates representing the squares directly bordering a
    central point (including those diagonally adjacent)

    Parameters:
        center: an n-dimensional tuple representing the coordinate of the point
        whose neighbors we are searching for
    """

    def get_permutations(tup, index):
        # Returns all permutations of tup where all indices after index (including index), are
        # raised by one, lowered by one, or stay the same.
        perms = []
        if index < len(tup):
            for i in [0, -1, 1]:
                base = list(tup)
                base[index] += i
                base = tuple(base)
                perms += get_permutations(base, index + 1)
        else:
            perms.append(tup)
        return perms

    neighbors = get_permutations(center, 0)
    del neighbors[0]  # Center doesn't count as its own neighbor. Luckily, its easy to remove, as the
    # first element returned by get_permutations is always the tuple that provided as input.
    return neighbors


def bounded_nd(values, lower, upper):
    """
    Returns true if a tuple of values is in between a specified upper and lower bound, or
    geometrically speaking, if the tuple is contained within the hyperrectangle defined by
    lower and upper.

    Parameters:
        values: a length-n tuple of numbers
        lower: a length-n tuple of values where each element represents the lower bound of
        the corresponding element in values
        upper: a length-n tuple of values where each element represents the upper bound of
        the corresponding element in values
    """

    for i in range(len(values)):
        if not lower[i] <= values[i] <= upper[i]:
            return False
    return True


def dig_nd(game, coordinate):
    """
    Recursively dig up square at coords and neighboring squares.

    Update the mask to reveal square at coord; then recursively reveal its
    neighbors, as long as coords does not contain and is not adjacent to a
    bomb.  Return a number indicating how many squares were revealed.  No
    action should be taken and 0 returned if the incoming state of the game
    is not 'ongoing'.

    The updated state is 'defeat' when at least one bomb is visible on the
    board after digging, 'victory' when all safe squares (squares that do
    not contain a bomb) and no bombs are visible, and 'ongoing' otherwise.

    Args:
       coordinate (tuple): Where to start digging

    Returns:
       int: number of squares revealed

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'mask': [[[False, False], [False, True], [False, False], [False, False]],
    ...               [[False, False], [False, False], [False, False], [False, False]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 3, 0))
    8
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    mask:
        [[False, False], [False, True], [True, True], [True, True]]
        [[False, False], [False, False], [True, True], [True, True]]
    state: ongoing
    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'mask': [[[False, False], [False, True], [False, False], [False, False]],
    ...               [[False, False], [False, False], [False, False], [False, False]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 0, 1))
    1
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    mask:
        [[False, True], [False, True], [False, False], [False, False]]
        [[False, False], [False, False], [False, False], [False, False]]
    state: defeat
    """

    if game['state'] == 'defeat' or game['state'] == 'victory':
        # If we're in a terminal state, we do nothing.
        return 0

    if get_element_nd(game['board'], coordinate) == '.':
        # If we hit a bomb, we lose. Massive bruh moment.
        set_element_nd(game['mask'], coordinate, True)
        game['state'] = 'defeat'
        return 1

    unrevealed_coords = set()  # A set of all hidden non-bomb squares
    lower = tuple([0] * len(game['dimensions']))  # The lower bound for any coordinate in our minefield
    upper = tuple([i - 1 for i in game['dimensions']])  # The upper bound for any coordinate in our minefield
    for square in nd_range(lower, upper):
        val = get_element_nd(game['board'], square)
        visible = get_element_nd(game['mask'], square)
        if not val == '.' and not visible:
            # If a non-bomb square is still hidden, we add it to our set of unrevealed coordinates
            unrevealed_coords.add(square)

    # Dig up non-bomb squares
    q = Queue(coordinate)
    revealed = 0
    while not q.is_empty():
        coord = q.next()
        if not get_element_nd(game['mask'], coord):
            set_element_nd(game['mask'], coord, True)
            revealed += 1

        if get_element_nd(game['board'], coord) == 0:
            neighbors = get_neighbors(coord)  # The coordinates neighboring coord (irrespective of grid bounds)
            # If our selected square borders no bombs, we reveal the surrounding squares.
            for neighbor in neighbors:
                if bounded_nd(neighbor, lower, upper) and not get_element_nd(game['mask'], neighbor) \
                        and neighbor not in q.get_history():
                    # If this neighboring non-bomb square has not already been revealed or queued for reveal,
                    # we add it to the queue
                    q.add(neighbor)

    if unrevealed_coords == q.history:
        # If we've revealed all hidden non-bomb squares this turn, declare victory
        game['state'] = 'victory'
    return revealed


class Queue:
    """
    Queue structure for search operations
    """

    def __init__(self, start=None):
        """
        Initialize Queue
        """
        self.newest = -1
        self.oldest = 0
        self.items = {}
        self.history = set()
        if start is not None:
            self.add(start)

    def next(self):
        """
        Returns the oldest element in the queue, then deletes it
        """
        # FIFO
        next_item = self.items.get(self.oldest, None)
        del self.items[self.oldest]
        # Will fail if dictionary is empty (I could put an if statement here but it would slow things down)
        self.oldest += 1
        return next_item

    def add(self, val):
        """
        Adds new value to the back of the queue
        """
        self.newest += 1
        self.items[self.newest] = val
        self.history.add(val)

    def is_empty(self):
        """
        Returns true if the queue is empty
        """
        return self.newest < self.oldest  # Probably slightly faster than len(self.items) == 0

    def get_history(self):
        """
        Returns a set containing any item that has ever been added to the queue
        """
        return self.history

    def get_items(self):
        """
        Returns a list of every item in the queue
        """
        return list(self.items.values())

    def print_items(self):
        """
        Prints a list of every item in the queue
        """
        print(list(self.items.values()))


def nd_range(lower, upper):
    """
    Returns all integer coordinates between lower and upper, or geometrically speaking,
    all points contained within the hyperrectangle defined by lower and upper.

    Parameters:
        lower: a length-n tuple of values where each element represents the lower bound of
        the corresponding element in values
        upper: a length-n tuple of values where each element represents the upper bound of
        the corresponding element in values
    """

    def get_permutations(tup, index):
        # Returns all permutations of tup where all indices after index (including index)
        # are between the corresponding elements of lower and upper
        perms = []
        if index < len(tup):
            for i in range(lower[index], upper[index] + 1):
                base = list(tup)
                base[index] += i
                base = tuple(base)
                perms += get_permutations(base, index + 1)
        else:
            perms.append(tup)
        return perms

    return get_permutations(lower, 0)


def render_nd(game, xray=False):
    """
    Prepare the game for display.

    Returns an N-dimensional array (nested lists) of '_' (hidden squares),
    '.' (bombs), ' ' (empty squares), or '1', '2', etc. (squares
    neighboring bombs).  The mask indicates which squares should be
    visible.  If xray is True (the default is False), the mask is ignored
    and all cells are shown.

    Args:
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    the mask

    Returns:
       An n-dimensional array of strings (nested lists)

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'mask': [[[False, False], [False, True], [True, True], [True, True]],
    ...               [[False, False], [False, False], [True, True], [True, True]]],
    ...      'state': 'ongoing'}
    >>> render_nd(g, False)
    [[['_', '_'], ['_', '3'], ['1', '1'], [' ', ' ']], [['_', '_'], ['_', '_'], ['1', '1'], [' ', ' ']]]

    >>> render_nd(g, True)
    [[['3', '.'], ['3', '3'], ['1', '1'], [' ', ' ']], [['.', '3'], ['3', '.'], ['1', '1'], [' ', ' ']]]
    """

    dims = game['dimensions']
    board_render = build_nd(dims, 0)  # Our rendered board
    lower = tuple([0] * len(dims))  # The lower bound for any coordinate in our minefield
    upper = tuple([i - 1 for i in dims])  # The upper bound for any coordinate in our minefield
    coords = nd_range(lower, upper)
    for square in coords:
        val = get_element_nd(game['board'], square)
        display = (' ' if val == 0 else str(val)) if (get_element_nd(game['mask'], square) or xray) else '_'
        set_element_nd(board_render, square, display)
    return board_render


if __name__ == "__main__":
    # Test with doctests. Helpful to debug individual lab.py functions.
    import doctest

    _doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    doctest.testmod(optionflags=_doctest_flags)  # runs ALL doctests

    # Alternatively, can run the doctests JUST for specified function/methods,
    # e.g., for render_2d or any other function you might want.  To do so, comment
    # out the above line, and uncomment the below line of code. This may be
    # useful as you write/debug individual doctests or functions.  Also, the
    # verbose flag can be set to True to see all test results, including those
    # that pass.
    #
    doctest.run_docstring_examples(dig_nd, globals(), optionflags=_doctest_flags, verbose=True)

