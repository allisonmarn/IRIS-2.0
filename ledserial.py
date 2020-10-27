import serial
import atexit
import os

__SERIAL = serial.Serial()

def __destructor():

    print('Closing LED Connection')

    __close()
    
atexit.register(__destructor)

def __connect(portno):
	__SERIAL.baudrate = 57600
	__SERIAL.port = 'COM'+str(portno)
	__SERIAL.open()

	
	
def __close():
	__SERIAL.close()
	

def __read_power():
	return __SERIAL.readline()


def connect(portno):
	__connect(portno)
	if __SERIAL.is_open:
		print('LEDs connected')

		
def read_power():
	return __read_power()

def close():
	return __close()
	
def send(command):
	#send command to the arduino
	__SERIAL.write(command.encode())
	
def receive(self):
	#receive the data from the arduino
	return __SERIAL.readline().decode('ascii')
	
