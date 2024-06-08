from bottle import get, post, run, request, response, hook, route
import random
from urllib.parse import quote, unquote
from game import get_greedy_action, state_to_int
import pickle
import json

# Load Q from the file
with open('Q.pkl', 'rb') as file:
    Q = pickle.load(file)

# Load best_Qs
with open('best_Qs.pkl', 'rb') as file:
    best_Qs = pickle.load(file)

#Current opponent
current_Q = Q

# Middleware to add CORS headers to the response
@hook('after_request')
def enable_cors():
    #should only allow all origins * in dev
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept, Authorization'

#needed to bypass cors for specific options req to endpoint
@route('/sendBoard', method='OPTIONS')
def send_board_options():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return {}

#end point for receiving board and sending new board with computer move
@post('/sendBoard')
def send_board():
    response.status = 200
    board = request.json

    action = get_greedy_action(current_Q, board, 0, 2)
    board[action] = 2
    return json.dumps(board)

#end point for changing opponent when game is over
@get('/gameOver')
def game_over():
    global current_Q
    response.status = 200
    best_Q = random.choice(best_Qs)
    current_Q = best_Q[0]
    print(best_Q[1])

run(host='localhost', port=5175, debug=True)

