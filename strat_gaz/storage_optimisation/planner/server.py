from flask import Flask
from flask import render_template

from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_socketio import send, emit

from flask_socketio import join_room, leave_room, rooms

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)


@app.route('/')
def sessions():
    return render_template('session.html')

@socketio.on('connect')
def handle_connect():
    send('Hello, you succesfully connected to the distributed computing planner of the scanner')

@socketio.on('message')
def handle_message(message):
    send('received' + message + '_' + str(rooms()))

@socketio.on('join')
def on_join(data):
    username = data['username']
    room = data['room']
    join_room(room)
    emit('message', username + ' has entered the room.', room=room)

@socketio.on('leave')
def on_leave(data):
    username = data['username']
    print("tip")
    room = data['room']
    leave_room(room)
    emit('message', username + ' has left the room.', room=room)

@socketio.on('request')
def handle_request(param):
    emit('data', 'info of the data , data returned by a function ', room =rooms()[0])
    #on écrit dans la db que le calcul est en cours associé à l'id de co


@socketio.on("computing")
def handle_computing(response):
    # on recupère le result
    # on écrit dans la db 
    # on marque la tache comme done
    emit('conclusion','sign that everything get good ')

class Planner:
    def __init__(self):
        pass

# @socketio.on('receive_msg')
# def handle_my_custom_event(json, methods=['GET', 'POST']):
#     print('received my event: ' + str(json))
#     socketio.emit('the_response', json)
    
if __name__ == '__main__':
    socketio.run(app, debug=True)
