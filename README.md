# A repository for:

- training agents to play tic-tac-toe with Q learning with epsilon-greedy strategy
- REST API to integrate with front-end

## Starting the bottle server:

- python3 integration.py

## Training and testing agents:

- python3 game.py

## How the training was done:

First, a pool of models was constructed. This was done by first training an agent against
a random opponent, and then continuosly training more models and adding them to the pool, until a pool of 50 models was reached. Then, 10 agents were trained at a time against opponents randomly selected from the pool, and the model that performed best in testing was added to the file best_Qs.pkl.

## Testing:

The testing of each trained model was done by playing 10 000 games against randomly selected opponents from old_Qs. Winning percentage was used as metric when choosing which model to add to best_Qs.

## Hyper parameters:

nbr_models = 10
nbr_ep = 100000
rewards = [10,-10,5,0]
alpha = 0.3
epsilon = 0.4
opponent_epsilon = 0.2
gamma = 0.9

## Files explained:

- old_Qs.pkl: pool consisting of 50 models that are used for training and testing
- Q.pkl: the current best model
- best_Qs: the best performing models from training against pool

## Integration:

I used bottle to implement a light REST API. The frontend sends a game board, and the API sends back a new board where the computer has made a move on the board.

## License

[MIT](https://choosealicense.com/licenses/mit/)
