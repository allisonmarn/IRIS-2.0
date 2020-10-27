import os
import atexit
from warnings import warn
from contextlib import suppress
import datetime
import PySpin
import numpy as np

__SYSTEM = PySpin.System.GetInstance()


__CAM = None


def __destructor():

    __cleanup_cam()

    if __SYSTEM.IsInUse():
        print('System is still in use')
    
atexit.register(__destructor)

def __cam_node_cmd(cam, cam_attr_str, cam_method_str, pyspin_mode_str=None, cam_method_arg=None):
    # Performs cam_method on input cam with optional access mode check
    # First, get camera attribute
    cam_attr = cam
    cam_attr_str_split = cam_attr_str.split('.')
    for sub_cam_attr_str in cam_attr_str_split:
        cam_attr = getattr(cam_attr, sub_cam_attr_str)

    # Print command info
    info_str = 'Executing: "' + '.'.join([cam_attr_str, cam_method_str]) + '('
    if cam_method_arg is not None:
        info_str += str(cam_method_arg)
    print(info_str + ')"')

    # Perform optional access mode check
    if pyspin_mode_str is not None:
        if cam_attr.GetAccessMode() != getattr(PySpin, pyspin_mode_str):
            raise RuntimeError('Access mode check failed for: "' + cam_attr_str + '" with mode: "' +
                               pyspin_mode_str + '".')

    # Format command argument in case it's a string containing a PySpin attribute
    if isinstance(cam_method_arg, str):
        cam_method_arg_split = cam_method_arg.split('.')
        if cam_method_arg_split[0] == 'PySpin':
            if len(cam_method_arg_split) == 2:
                cam_method_arg = getattr(PySpin, cam_method_arg_split[1])
            else:
                raise RuntimeError('Arguments containing nested PySpin arguments are currently not '
                                   'supported...')

    # Perform command
    if cam_method_arg is None: #pylint: disable=no-else-return
        return getattr(cam_attr, cam_method_str)()
    else:
        return getattr(cam_attr, cam_method_str)(cam_method_arg)

def __cleanup_cam():
    # cleans up camera
    global __CAM

    # End acquisition and de-init
    with suppress(Exception):
        end_acquisition()
    with suppress(Exception):
        __CAM.DeInit()

    # Clear camera reference
    __CAM = None
	

def __find_cam(cam_serial):
    #returns camera object 

    # Retrieve cameras from the system
    cam_list = __SYSTEM.GetCameras()
	
    num_cameras = cam_list.GetSize()
	
    if num_cameras == 1:
        cam_found = cam_list[0]
    else:
        print("More than one camera connected")

    return cam_found
	
def __get_image(cam):

    # Get image object
    image = cam.GetNextImage()
	
    # Initialize image dict
    image_dict = {}

    # Ensure image is complete
    if not image.IsIncomplete():
        # Get data/metadata
        image_dict['data'] = image.GetNDArray()
        image_dict['timestamp'] = image.GetTimeStamp()
        image_dict['bitsperpixel'] = image.GetBitsPerPixel()
    image.Release()
    return image_dict
	
def __get_image_and_avg(cam,num_to_avg):
	# Gets images and info from camera
	try:
		for i in range(0,num_to_avg):
			# Get image object
			image = cam.GetNextImage()

			# Initialize image dict
			image_dict = {}

			# Ensure image is complete
			if not image.IsIncomplete():
			# Get data/metadata
				if i == 0:
					img_array=np.array(image.GetNDArray(),dtype=np.float)
					arr = img_array
				else:
					img_array=np.array(image.GetNDArray(),dtype=np.float)
					arr=arr+img_array
			else:
				print('Incomplete Image')
		image.Release()
		image_dict['data'] = arr/num_to_avg
		image_dict['timestamp'] = image.GetTimeStamp()
		print('Averaged Frame: '+str(datetime.datetime.now()))
		image_dict['bitsperpixel'] = image.GetBitsPerPixel()
	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		return False
	except RuntimeError:
		print(sys.exc_info()[0])

        
	return image_dict
	
def __init_cam(cam):
	# Init() camera
	cam.Init()
			 
def __validate_cam(cam, cam_str):
    # Checks to see if camera is valid

    if not cam.IsValid():
        raise RuntimeError('Camera is not valid.')

def __validate_cam_init(cam, cam_str):
    # Checks to see if camera is valid and initialized
    __validate_cam(cam, cam_str)

    if not cam.IsInitialized():
        raise RuntimeError('Camera is not initialized.')

def __validate_cam_streaming(cam, cam_str):
    # Checks to see if camera is valid, initialized, and streaming

    __validate_cam_init(cam, cam_str)

    if not cam.IsStreaming():
        raise RuntimeError(cam_str + ' cam is not streaming. Please start_acquisition() it.')

def __get_and_validate_init_cam():
    # Validates initialization and returns it

    cam = __get_cam()
    __validate_cam_init(cam, 'Camera')
    return cam

def __get_and_validate_streaming_cam():
    # Validates streaming of camera then returns it

    cam = __get_cam()
    __validate_cam_streaming(cam, 'Camera')
    return cam	
	
def __get_cam():
    # Returns Camera

    if __CAM is None:
        raise RuntimeError('Camera not found')

    return __CAM
	
def cam_node_cmd(cam_attr_str, cam_method_str, pyspin_mode_str=None, cam_method_arg=None):

    return __cam_node_cmd(__get_and_validate_init_cam(),
                          cam_attr_str,
                          cam_method_str,
                          pyspin_mode_str,
                          cam_method_arg)

def end_acquisition():
    # Ends acquisition
    __get_and_validate_streaming_cam().EndAcquisition()

def find_cam(cam_serial):
    # Finds Camerade
    global __CAM 

    cam = __find_cam(cam_serial)

    # Cleanup AFTER new camera is found successfully
    __cleanup_cam()

    # Assign camera
    __CAM = cam
    return __CAM


def disable_auto_exp():
	print('Disabling Auto Exposure')	
	cam_node_cmd('ExposureAuto','SetValue','RW',PySpin.ExposureAuto_Off)
					 				 
def disable_auto_frame():
    global __CAM
    nodemap = __CAM.GetNodeMap()

    node_acquisition_en = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnabled'))
    if not PySpin.IsAvailable(node_acquisition_en) or not PySpin.IsWritable(node_acquisition_en):
        print('Unable to set enable acquisition frame rate. Aborting...')
        return False
    node_acquisition_en.SetValue(True)
    print('Enabling frame rate control')
	
    node_acquisition_auto = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionFrameRateAuto'))
    if not PySpin.IsAvailable(node_acquisition_auto) or not PySpin.IsWritable(node_acquisition_auto):
        print('Unable to turn off Auto Frame Rate. Aborting...')
        return False
    node_acquisition_auto_off = node_acquisition_auto.GetEntryByName('Off')
    if not PySpin.IsAvailable(node_acquisition_auto_off) or not PySpin.IsReadable(node_acquisition_auto_off):
        print('Unable to turn off Auto Frame Rate. Aborting...')
        return False
    node_acquisition_auto.SetIntValue(node_acquisition_auto_off.GetValue())
    
def disable_auto_gain():
	print('Disabling Auto Gain')
	cam_node_cmd('GainAuto','SetValue','RW',PySpin.GainAuto_Off)

def get_exp_max():
	 return cam_node_cmd('ExposureTime','GetMax')
					 
def get_exp_min():
	 return cam_node_cmd('ExposureTime','GetMin')

def get_fps_max():
    return cam_node_cmd('AcquisitionFrameRate','GetMax')
    
def get_fps_min():
	return cam_node_cmd('AcquisitionFrameRate','GetMin')
	
def get_frame_rate():
    return cam_node_cmd('AcquisitionFrameRate','GetValue')		

def set_gamma(gamma_val):
	print('Setting Gamma to '+ str(gamma_val))
	cam_node_cmd('Gamma','SetValue','RW',gamma_val)

def get_image():
    # Gets image from camera 
    return __get_image(__get_and_validate_streaming_cam())

def get_image_and_avg(num_to_avg):
    # Gets and averages images from camera 
	return __get_image_and_avg(__get_and_validate_streaming_cam(),num_to_avg)

def init_cam():
	# Initializes camera		 
	__init_cam(__get_cam())

def set_exposure(exposure):
	cam_node_cmd('ExposureTime','SetValue','RW',exposure)
		 
def set_frame_rate(frame_rate):
    cam_node_cmd('AcquisitionFrameRate','SetValue','RW',frame_rate)
    
def set_gain(gain):
	cam_node_cmd('Gain','SetValue','RW',gain)

def set_pixel_format(format):
    global __CAM
    try:
        nodemap = __CAM.GetNodeMap()
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):

            # Retrieve the desired entry node from the enumeration node
            node_pixel_format_mono16 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono16'))
            if PySpin.IsAvailable(node_pixel_format_mono16) and PySpin.IsReadable(node_pixel_format_mono16):

                # Retrieve the integer value from the entry node
                pixel_format_mono16 = node_pixel_format_mono16.GetValue()

                # Set integer as new value for enumeration node
                node_pixel_format.SetIntValue(pixel_format_mono16)
                print('Setting format to ' + format)
    except:
        print('Cannot set pixel format')
        
def set_video_mode(mode):
    global __CAM

    nodemap = __CAM.GetNodeMap()
    node_video_mode = PySpin.CEnumerationPtr(nodemap.GetNode("VideoMode"))
    if not PySpin.IsAvailable(node_video_mode) or not PySpin.IsWritable(node_video_mode):
        print('Unable to set VideoMode. Aborting...')
        return False
		
    node_video_mode_7 = node_video_mode.GetEntryByName("Mode7")
    if not PySpin.IsAvailable(node_video_mode_7) or not PySpin.IsReadable(node_video_mode_7):
        print('Unable to set VideoMode to Mode7 (entry retrieval). Aborting...')
        return False

    video_mode_7 = node_video_mode_7.GetValue()
    node_video_mode.SetIntValue(video_mode_7)
    print('Video Mode set to Mode '+ mode)

def start_acquisition():
    # Starts acquisition
    __get_and_validate_init_cam().BeginAcquisition()

	
	