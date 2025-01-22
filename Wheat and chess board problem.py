import numpy as np
import matplotlib.pyplot as plt

# Create a 2x2 board for the chessboard
board_2x2 = np.array([[1, 2], [4, 8]])
print("Grains of wheat on a 2x2 chessboard:")
print(board_2x2)

#
def wheat_on_board(n, m):
    return np.array([[2**(i + j) for j in range(m)] for i in range(n)])

# 8x8 board
board_8x8 = wheat_on_board(8, 8)
print("Grains of wheat on an 8x8 chessboard:")
print(board_8x8)

# Total number of grains
total_wheat = np.sum(board_8x8)
print(f"Total number of grains on an 8x8 chessboard: {total_wheat}")

# Average per column
average_per_column = np.mean(board_8x8, axis=0)
print(f"Average number of grains per column: {average_per_column}")

# Bar chart
plt.xlabel("Column")
plt.ylabel("Number")
plt.title("Number of grains per column")
plt.bar(np.arange(1, 9), average_per_column)
plt.show()

#

# Heatmap
plt.xlabel("Column")
plt.ylabel("Row")
plt.title("Heatmap of wheat grains")
plt.pcolor(board_8x8, cmap='plasma')
plt.colorbar()
plt.show()


first_half = board_8x8[:4, :]
second_half = board_8x8[4:, :]

# Compare totals
first_half_total = np.sum(first_half)
second_half_total = np.sum(second_half)
ratio = second_half_total / first_half_total

print(f"The second half of the chessboard has {ratio} times more wheat than the first half.")
#

def wheat_with_append(n, m):
    board = np.array([1])
    for _ in range(n * m - 1):
        board = np.append(board, 2 * board[-1])
    return board.reshape(n, m)

board_with_append = wheat_with_append(8, 8)
print("Grains of wheat with np.append():")
print(board_with_append)


def wheat_with_broadcast(n, m):
    indices = np.arange(n * m)
    return (2**indices).reshape(n, m)

board_with_broadcast = wheat_with_broadcast(8, 8)
print("Grains of wheat with broadcast:")
print(board_with_broadcast)


%%timeit
wheat_with_append(8, 8)

%%timeit
wheat_with_broadcast(8, 8)
