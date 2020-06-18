import socketio
import time





sio = socketio.Client() 
@sio.event
def connect():
    print("I'm connected!")
@sio.event
def connect_error():
    print("The connection failed!")
@sio.event
def disconnect():
    print("I'm disconnected!")

@sio.on("message")
def handle_message(msg):
    print(msg)

@sio.on('data')
def handle_data(data):
    #optimization
    print('optimisé')
    sio.emit('computing', 'rep')

def main():
    sio.connect('http://127.0.0.1:5000/')
    sid = sio.sid
    print('my sid is', sio.sid)
    sio.emit('request',"param")
    time.sleep(3)
    sio.disconnect()

main()