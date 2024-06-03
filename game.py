import numpy as np
import array as arr
from typing import List
import random
import pickle

#0 = no brick
#1 = model brick
#2 = old model brick

def print_board(board):
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    for i in range(3):
        print('|'.join(symbols[board[i * 3 + j]] for j in range(3)))
        if i < 2:
            print('-' * 5)


# returns index of next move
def random_model(board: List[int] ):
    found = False
    while(not found):
        random_integer = random.randint(0, len(board) - 1)
        if(board[random_integer] == 0):
            found = True
        
    return random_integer

#check if player(either 1 or 2) has won
def has_won(board: List[int], player: int):
    split_board = np.array_split(board, 3)
    
    # Check rows
    for row in split_board:
        if row[0] == player and row[1] == player and row[2] == player:
            return True
        
    # Check columns
    for col in range(3):
        if split_board[0][col] == player and split_board[1][col] == player and split_board[2][col] == player:
            return True
        
    # Check diagonals
    if split_board[0][0] == player and split_board[1][1] == player and split_board[2][2] == player:
        return True
    if split_board[0][2] == player and split_board[1][1] == player and split_board[2][0] == player:
        return True
    
    return False
    
#reward function, returns reward based on board and list of rewards
def reward(board: List[int], rewards: List[int]):
    #reward for winning
    if(has_won(board, 1)):
        return rewards[0]
    #reward for losing
    elif(has_won(board, 2)):
        return rewards[1]
    #reward for tie
    elif(len(get_valid_actions(board)) == 0):
        return rewards[2]
    #ongoing reward
    else:
        return rewards[3]
    

# we use the current state of the board as the state
# in this function we convert the state of the board to a unique integer
# which can be used to index q matrix
def state_to_int(board):
    return int(''.join(map(str, board)), 3)

#get valid actions
def get_valid_actions(board):
    return [i for i,cell in enumerate(board) if cell == 0]

# maskes Q-values so unvalid actions are set to -inf, returns argmax
def masked_next_action(Q,state,valid_actions):
    #gets all q values for a given state
    state_q_values = Q[state]
    #create masked q-values [-inf,...-inf]
    masked_q_values = np.full(state_q_values.shape, -np.inf)
    #turns on available action values
    masked_q_values[valid_actions] = state_q_values[valid_actions]
    #print(masked_q_values)
    return np.argmax(masked_q_values)

#select greedy action
def get_greedy_action(Q,board,epsilon):
    state = state_to_int(board)
    valid_actions = get_valid_actions(board)
    if (len(valid_actions) == 1):
        return valid_actions[0]
    if random.random() < epsilon and len(valid_actions) > 0:
        return random.choice(valid_actions)
    else:            
        return masked_next_action(Q,state,valid_actions)
    
def q_learning_updates(Q, state, action, reward, next_state, alpha, gamma,terminate,next_board):
    if(not terminate):
        valid_actions_next = get_valid_actions(next_board)
        best_next_action = masked_next_action(Q,next_state,valid_actions_next)
        #best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action]
    else:
        td_target = reward
    td_error = td_target - Q[state][action]
    Q[state][action] += alpha * td_error

 #testing help methods

#training

#hyperparams
#number of episodes
nbr_ep = 20000
#initilize empty Q
Q = np.zeros((3**9, 9))
#initilize rewards
rewards = [10,-10,5,0]
#learning rate
alpha = 0.5
#explore random
epsilon = 0.3
#penalty
gamma = 0.5


#Switch starting order after each episode
starting_order = False

#Keep track of beginning of episode


# training loop

#only run if executed directly, trains and tests agent
if __name__ == "__main__":
    for i in range(nbr_ep):
        n = 0
        board = [0,0,0,0,0,0,0,0,0]
        while True:
            #change starting order between episodes
            if(n == 0 and starting_order):
                board[random_model(board)] = 2

            #current state
            state = state_to_int(board)

            #immediate reward
            immediate_reward = reward(board, rewards)

            #action selected, use get_greedy_action
            action = get_greedy_action(Q,board,epsilon)
            #action = np.argmax(Q[state])

            #if not terminate state
            if (immediate_reward == rewards[3]):
                #Next state is determined
                #next state is when both parts has played
                #current state -> m1 move with certain action -> m2 move -> next state
                board[action] = 1
                
                #have to check if board is full or already won before old model moves
                if(len(get_valid_actions(board)) >= 1 and not has_won(board, 1)):
                    board[random_model(board)] = 2
                
                #next state is derived
                next_state = state_to_int(board)
                q_learning_updates(Q,state, action, immediate_reward, next_state, alpha, gamma, False,board)
                n += 1
            else:
                #episode complete
                q_learning_updates(Q,state, action, immediate_reward, None, alpha, gamma, True, board)
                starting_order = not starting_order
                break


    #Test

    #some parameters
    nbr_of_tests = 10000
    result = [0,0,0]
    test_starting_order = False

    #based on recieved reward, returns terminate(true or false) and result
    def check_termination(r):
        terminate = False
        result = 0
        if r == rewards[0]:  # New model wins
            terminate = True
            result = 0
        elif r == rewards[1]:  # Old model wins
            terminate = True
            result = 1
        elif r == rewards[2]:  # Tie
            terminate = True
            result = 2
        return [terminate,result]

    for i in range(nbr_of_tests):
        # print("Test", i + 1)
        # print_board(board)
        test_starting_order = not test_starting_order
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        n = 0

        while True:
            # print_board(board)
            # print("\n")
            # Check if the game should terminate based on the current board
            current_reward = reward(board, rewards)
            check = check_termination(current_reward)
            if check[0]:
                result[check[1]] += 1
                break

            # Perform actions based on the starting order and turn
            if test_starting_order:
                if n % 2 == 0:
                    action = get_greedy_action(Q, board, 0)
                    board[action] = 1
                else:
                    if get_valid_actions(board):
                        board[random_model(board)] = 2
            else:
                if n % 2 == 0:
                    if get_valid_actions(board):
                        board[random_model(board)] = 2
                else:
                    action = get_greedy_action(Q, board, epsilon)
                    board[action] = 1

            # Increment turn counter
            n += 1

    with open('Q.pkl', 'wb') as file:
        pickle.dump(Q, file)
    print("Number of Tests:", nbr_of_tests)
    print("New Model Wins:", result[0])
    print("Old Model Wins:", result[1])
    print("Ties:", result[2])
