#idea
# I have the trained Q matrix, and the get_greedy_action(Q, board, 0)
#Game loop:
#vue gives board in [0,0,0,0,0,0,0,0,0] form
#python returns action index and sends to vue, get_greedy_action(Q,board,0)
#vue changes action index to fit 2d array
#computer makes move: computer.value[index[y]][x]]

#left on training model:
#train multiple models, change old model when better test result