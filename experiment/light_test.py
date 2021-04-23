import socket
from socketIO_client import SocketIO, BaseNamespace
import asyncio
from threading import Thread
import random
import time

choices = [0,4095,2500,1000]
def LIGHT_VALUES():
    for i in range(len(input_vals)):
        input_vals[i]=random.choice(choices)
    return input_vals
#LIGHT_VALUES = [ [4095] * 16, [0] * 16] # flip between on and off

EVOLVER_IP = '192.168.1.2'
EVOLVER_PORT = 8081
EVOLVER_NS = None
socketIO = None

class EvolverNamespace(BaseNamespace):
    def on_connect(self, *args):
        print("Connected to eVOLVER as client")

    def on_disconnect(self, *args):
        print("Discconected from eVOLVER as client")

    def on_reconnect(self, *args):
        print("Reconnected to eVOLVER as client")

    def on_broadcast(self, data):
        print(data)

def run_client():
    global EVOLVER_NS, socketIO
    socketIO = SocketIO(EVOLVER_IP, EVOLVER_PORT)
    EVOLVER_NS = socketIO.define(EvolverNamespace, '/dpu-evolver')
    socketIO.wait()

def run_test(time_to_wait, selection):
    time.sleep(time_to_wait)
    print('Sending data...')
    # Send light commands
    data = {'param': 'light', 'value': LIGHT_VALUES(), 'immediate': True}
    print(data)
    EVOLVER_NS.emit('command', data, namespace = '/dpu-evolver')
    # Set things for the next one
    selection = 1 - selection
    time_to_wait = 1 #random.randint(1,5)
    print('Seconds to wait: ' + str(time_to_wait))
    run_test(time_to_wait, selection)

def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


if __name__ == '__main__':
    try:
        new_loop = asyncio.new_event_loop()
        t = Thread(target = start_background_loop, args = (new_loop,))
        t.daemon = True
        t.start()
        new_loop.call_soon_threadsafe(run_client)
        time.sleep(5)
        run_test(0, 0)
    except KeyboardInterrupt:
        socketIO.disconnect()
