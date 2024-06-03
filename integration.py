#idea
# I have the trained Q matrix, and the get_greedy_action(Q, board, 0)
#Game loop:
#vue gives board in [0,0,0,0,0,0,0,0,0] form
#python returns action index and sends to vue, get_greedy_action(Q,board,0)
#vue changes action index to fit 2d array
#computer makes move: computer.value[index[y]][x]]

# other solution:
# send the whole matrix, compute on client. Pros: only one request. Cons: have to make logic in vue

#left on training model:
#train multiple models, change old model when better test result

from bottle import get, post, run, request, response, hook, route

from urllib.parse import quote, unquote

from game import get_greedy_action, state_to_int

import json

# Middleware to add CORS headers to the response
@hook('after_request')
def enable_cors():
    #should only allow all origins * in dev
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept, Authorization'


@get('/ping')
def ping():

    response.status = 200
    return {"pong": "pong"}

@route('/sendBoard', method='OPTIONS')
def send_board_options():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return {}

@post('/sendBoard')
def send_board():
    response.status = 200
    print(request.json)
    return {"sucess" : "yes"}

run(host='localhost', port=5175, debug=True)

