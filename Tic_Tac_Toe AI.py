# preparing the game board
def MakeBoard(board):
    print(board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('- + - + -')
    print(board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('- + - + -')
    print(board[7] + ' | ' + board[8] + ' | ' + board[9])
    print("\n")

# checking if a position is empty
def CheckEmpty(position):
    if board[position] == ' ':
        return True
    else:
        return False

# inserting the moves to the board
def InsertMove(letter, position):
    if CheckEmpty(position):
        board[position] = letter
        MakeBoard(board)
        if (Draw()):
            print("Draw!")
            exit()
        if Wining():
            if letter == 'X':
                print("Bot wins!")
                exit()
            else:
                print("Player wins!")
                exit()

        return
    else:
        print("Can't insert there!")
        position = int(input("Please enter new position:  "))
        InsertMove(letter, position)
        return

# checking if a win happen after each turn
def Wining():
    if (board[1] == board[2] and board[1] == board[3] and board[1] != ' '):
        return True
    elif (board[4] == board[5] and board[4] == board[6] and board[4] != ' '):
        return True
    elif (board[7] == board[8] and board[7] == board[9] and board[7] != ' '):
        return True
    elif (board[1] == board[4] and board[1] == board[7] and board[1] != ' '):
        return True
    elif (board[2] == board[5] and board[2] == board[8] and board[2] != ' '):
        return True
    elif (board[3] == board[6] and board[3] == board[9] and board[3] != ' '):
        return True
    elif (board[1] == board[5] and board[1] == board[9] and board[1] != ' '):
        return True
    elif (board[7] == board[5] and board[7] == board[3] and board[7] != ' '):
        return True
    else:
        return False

# checking who win in the minimax function
def Who_Won(mark):
    if board[1] == board[2] and board[1] == board[3] and board[1] == mark:
        return True
    elif (board[4] == board[5] and board[4] == board[6] and board[4] == mark):
        return True
    elif (board[7] == board[8] and board[7] == board[9] and board[7] == mark):
        return True
    elif (board[1] == board[4] and board[1] == board[7] and board[1] == mark):
        return True
    elif (board[2] == board[5] and board[2] == board[8] and board[2] == mark):
        return True
    elif (board[3] == board[6] and board[3] == board[9] and board[3] == mark):
        return True
    elif (board[1] == board[5] and board[1] == board[9] and board[1] == mark):
        return True
    elif (board[7] == board[5] and board[7] == board[3] and board[7] == mark):
        return True
    else:
        return False

# checking if a draw happens
def Draw():
    for key in board.keys():
        if (board[key] == ' '):
            return False
    return True

# allow the player to put his move
def playerMove():
    position = int(input("Enter the position for 'O':  "))
    InsertMove(player, position)
    return

# allow the bot to put his move
def BotMove():
    bestScore = -1000
    bestMove = 0
    for key in board.keys():
        if (board[key] == ' '):
            board[key] = bot
            score = minimax(board, 0, False)
            board[key] = ' '
            if (score > bestScore):
                bestScore = score
                bestMove = key

    InsertMove(bot, bestMove)
    return

# minimax function that allow the bot to always win
def minimax(board, depth, isMaximizing):
    if (Who_Won(bot)):
        return 1
    elif (Who_Won(player)):
        return -1
    elif (Draw()):
        return 0

    if (isMaximizing):
        bestScore = -1000
        for key in board.keys():
            if (board[key] == ' '):
                board[key] = bot
                score = minimax(board, depth + 1, False)
                board[key] = ' '
                if (score > bestScore):
                    bestScore = score
        return bestScore
    else:
        bestScore = 1000
        for key in board.keys():
            if (board[key] == ' '):
                board[key] = player
                score = minimax(board, depth + 1, True)
                board[key] = ' '
                if (score < bestScore):
                    bestScore = score
        return bestScore


board = {1: ' ', 2: ' ', 3: ' ',
         4: ' ', 5: ' ', 6: ' ',
         7: ' ', 8: ' ', 9: ' '}

MakeBoard(board)
print("Bot goes first! Good luck.")
print("Positions are as follow:")
print("1, 2, 3 ")
print("4, 5, 6 ")
print("7, 8, 9 ")
print("\n")
player = 'O'
bot = 'X'

while not Wining():
    BotMove()
    playerMove()