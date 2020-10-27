import sys
from multiprocessing import Queue
import functools
import datetime
import serial
from tkinter import messagebox
from tkinter import filedialog

import numpy as np


import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

import multiprocessing as mp
import tifffile as tf

# files to include
import spincam
import ledserial_proto as ledserial

# Camera Properties Min/Max
__FPS_MIN = 1
__FPS_MAX = 67           
__GAIN_MIN = 0           
__GAIN_MAX = 47          # Units are dB
__EXPOSURE_MIN = 0.006       # microseconds
__EXPOSURE_MAX = 2999999 # microseconds seconds
__BVAL=0
__GVAL=0  

# GUI params
__QUEUE = Queue()
__STREAM = False
__IMSHOW_DICT = {'imshow': None, 'imshow_size': None, 'max_val': None}
__HIST_DICT = {'bar': None, 'max_val': None}
__GUI_DICT = None
__PD = 0
__COM_PORT =5
__SAVE=0
__CAM= None
p = None
__BITS=0


def __find_and_init_cam(_=None):
	#Finds and initializes camera
	global __EXPOSURE_MAX
	global __EXPOSURE_MIN
	global __FPS_MAX
	global __FPS_MIN

	find_and_init_text = __GUI_DICT['display_dict']['find_and_init_text'].text
	print('Connecting camera...')
	spincam.find_cam(find_and_init_text)
	spincam.init_cam()
	__EXPOSURE_MIN = spincam.get_exp_min()
	__EXPOSURE_MAX = spincam.get_exp_max()
	__FPS_MIN = spincam.get_fps_min()
	__FPS_MAX = spincam.get_fps_max()
	spincam.disable_auto_exp()
	spincam.disable_auto_gain()
	__init_gain(0)
	spincam.set_gamma(1)
	spincam.set_pixel_format('Mono16')
	spincam.set_video_mode('7')
	ledserial.connect(__COM_PORT)

	
def __choose_directory(_=None):
    dir = filedialog.askdirectory()
    __GUI_DICT['directory_text'].set_val(dir)
	
def __start_stream(_=None):
    # Starts stream of cameras
    global __STREAM
	
    # Ensure aren't already streaming
    if not __STREAM:
        print('Starting stream...')

		# Set buffer to newest only
        spincam.cam_node_cmd('TLStream.StreamBufferHandlingMode',
                                       'SetValue',
                                       'RW',
                                       'PySpin.StreamBufferHandlingMode_NewestOnly')

        # Set acquisition mode to continuous
        spincam.cam_node_cmd('AcquisitionMode',
                                       'SetValue',
                                       'RW',
                                       'PySpin.AcquisitionMode_Continuous')

        # Start acquisition
        spincam.start_acquisition()

        # Enable stream
        __STREAM = True
		
	
def __stop_stream(_=None):
    # Stops stream of cameras
    global __STREAM

    # Make sure they're streaming
    if __STREAM:
        print('Stopping stream...')

        # Stop acquisition
        spincam.end_acquisition()
        # End stream
        __STREAM = False	
		
def __plot_image(image, max_val, image_axes, imshow_dict):
    # plots image

    # If image size changes or max val changes, replot imshow
    if image.shape != imshow_dict['imshow_size'] or max_val != imshow_dict['max_val']:
        image_axes.cla()
        imshow_dict['imshow'] = image_axes.imshow(image, cmap='gray', vmin=0, vmax=max_val)
        imshow_dict['imshow_size'] = image.shape
        imshow_dict['max_val'] = max_val
        image_axes.set_xticklabels([])
        image_axes.set_yticklabels([])
        #image_axes.set_xticks(list(range(1,2736)))
        #image_axes.set_yticks(list(range(1,2192)))
    else:
        # Can just "set_data" since data is the same size and has the same max val
        imshow_dict['imshow'].set_data(image)

    return imshow_dict
	
def __plot_hist(image, max_val, hist_axes, hist_dict):
    # plots histogram

    # Calculate histogram
    num_bins = 100
    hist, bins = np.histogram(image.ravel(), bins=num_bins, range=(0, max_val))

    # If no histogram or max value changes, replot histogram
    if hist_dict['bar'] is None or hist_dict['max_val'] != max_val:
        # Reset axes and plot hist
        hist_axes.cla()
        hist_dict['bar'] = hist_axes.bar(bins[:-1], hist, color='k', width=(max_val+1)/num_bins)
        hist_dict['max_val'] = max_val
        hist_axes.set_xticklabels([])
        hist_axes.set_yticklabels([])
        hist_axes.set_xticks([])
        hist_axes.set_yticks([])
    else:
        # Reset height
        for i, bar in enumerate(hist_dict['bar']): # pylint: disable=blacklisted-name
            bar.set_height(hist[i])

    return hist_dict

def __plot_image_and_hist(image, max_val, image_axes, imshow_dict, hist_axes, hist_dict): # pylint: disable=too-many-arguments
    # plots image and histogram

    # Plot image
    imshow_dict = __plot_image(image, max_val, image_axes, imshow_dict)
    # Plot histogram
    hist_dict = __plot_hist(image, max_val, hist_axes, hist_dict)

    return (imshow_dict, hist_dict)

def __cam_plot(fig, pos, cam_str, options_height, padding):
    # Creates 'camera' plot

    # Set position params
    num_options = 1
    residual_height = pos[3]-(3+num_options)*padding-num_options*options_height
    image_height = residual_height*0.85
    image_width = pos[2]-2*padding
    hist_height = residual_height-image_height

    # Set axes
    image_pos = [pos[0]+padding, pos[1]+pos[3]-image_height-padding, image_width, image_height]
    image_axes = fig.add_axes(image_pos)
    image_axes.set_xticklabels([])
    image_axes.set_yticklabels([])
    image_axes.set_xticks([])
    image_axes.set_yticks([])

    hist_pos = [image_pos[0], image_pos[1]-hist_height-padding, image_width, hist_height]
    hist_axes = fig.add_axes(hist_pos)
    hist_axes.set_xticklabels([])
    hist_axes.set_yticklabels([])
    hist_axes.set_xticks([])
    hist_axes.set_yticks([])

    find_and_init_button_pos = [image_pos[0],
                                hist_pos[1]-options_height-padding,
                                (image_width-padding)*0.5,
                                options_height]
    find_and_init_button_axes = fig.add_axes(find_and_init_button_pos)

    find_and_init_text_pos = [find_and_init_button_pos[0]+find_and_init_button_pos[2]+padding,
                              find_and_init_button_pos[1],
                              (image_width-padding)*0.5,
                              options_height]
    find_and_init_text_axes = fig.add_axes(find_and_init_text_pos)
	
	

    # Set widgets
    find_and_init_button = Button(find_and_init_button_axes, 'Find and Init ' + cam_str)
    find_and_init_button.label.set_fontsize(7)
    find_and_init_text = TextBox(find_and_init_text_axes, '')

    return {'image_axes': image_axes,
            'hist_axes': hist_axes,
            'find_and_init_button': find_and_init_button,
            'find_and_init_text': find_and_init_text}

def __init_gain(gain):
    # gain text callback
	
	# Set gain for camera
    print('Initializing Gain to ' + str(gain))
    spincam.set_gain(gain)
    # Update text
    #__GUI_DICT['gain_text'].eventson = False
    #__GUI_DICT['gain_text'].set_val(gain)
    #__GUI_DICT['gain_text'].eventson = True

    # Update slider to match text
    #__GUI_DICT['gain_slider'].eventson = False
    #__GUI_DICT['gain_slider'].set_val(gain)
    #__GUI_DICT['gain_slider'].eventson = True
		
def __fps_slider(_=None):
    # fps slider callback

    framerate = __GUI_DICT['fps_slider'].val

    # Update text to match slider
    __GUI_DICT['fps_text'].eventson = False
    __GUI_DICT['fps_text'].set_val(framerate)
    __GUI_DICT['fps_text'].eventson = True

    # Set fps for camera
    framerate=min(__FPS_MAX,framerate)
    framerate=max(__FPS_MIN,framerate)
    spincam.set_frame_rate(framerate)
    fr=spincam.get_frame_rate()
    print('Frame Rate is set to '+ str(fr))
	
def __fps_text(_=None):
    # fps text callback

    fps_text = __GUI_DICT['fps_text'].text
    if not fps_text:
        return

    framerate = float(fps_text)

    # Update slider to match text
    __GUI_DICT['fps_slider'].eventson = False
    __GUI_DICT['fps_slider'].set_val(framerate)
    __GUI_DICT['fps_slider'].eventson = True

	# Set fps for camera
    framerate=min(__FPS_MAX,framerate)
    framerate=max(__FPS_MIN,framerate)
    spincam.set_frame_rate(framerate)
    fr=spincam.get_frame_rate()
    print('Frame Rate is set to '+ str(fr))
	
def __gain_slider(_=None):
    # gain slider callback

    gain = __GUI_DICT['gain_slider'].val

    # Update text to match slider
    __GUI_DICT['gain_text'].eventson = False
    __GUI_DICT['gain_text'].set_val(gain)
    __GUI_DICT['gain_text'].eventson = True

    # Set gain for camera
    spincam.set_gain(gain)
	
def __gain_text(_=None):
    # gain text callback

    gain_text = __GUI_DICT['gain_text'].text
    if not gain_text:
        return

    gain = float(gain_text)

    # Update slider to match text
    __GUI_DICT['gain_slider'].eventson = False
    __GUI_DICT['gain_slider'].set_val(gain)
    __GUI_DICT['gain_slider'].eventson = True

	# Set gain for camera
    spincam.set_gain(gain)
			
def __slider_with_text(fig, pos, slider_str, val_min, val_max, val_default, padding): # pylint: disable=too-many-arguments
    # Creates a slider with text box

    # Set position params
    slider_padding = 0.1
    slider_text_width = (0.5-3*padding)/3

    # Slider
    slider_pos = [pos[0]+slider_padding+padding,
                  pos[1],
                  pos[2]-slider_padding-3*padding-slider_text_width,
                  pos[3]]
    slider_axes = fig.add_axes(slider_pos)
    slider = Slider(slider_axes,
                    slider_str,
                    val_min,
                    val_max,
                    valinit=val_default,
                    dragging=False)
    slider.label.set_fontsize(7)
    slider.valtext.set_visible(False)

    # Text
    text_pos = [slider_pos[0]+slider_pos[2]+padding,
                slider_pos[1],
                slider_text_width,
                pos[3]]
    text_axes = fig.add_axes(text_pos)
    text = TextBox(text_axes, '')

    return (slider, text)

def __exposure_slider(_=None):
    # Exposure slider callback
    global __EXPOSURE_MAX
    global __EXPOSURE_MIN
    exposure = __GUI_DICT['exposure_slider'].val

    # Update text to match slider
    __GUI_DICT['exposure_text'].eventson = False
    __GUI_DICT['exposure_text'].set_val(exposure)
    __GUI_DICT['exposure_text'].eventson = True

    # Set exposure for cameras
    exposure = min(__EXPOSURE_MAX,exposure)
    exposure = max(__EXPOSURE_MIN,exposure)
    spincam.set_exposure(exposure)

def __exposure_text(_=None):
    # Exposure text callback
    global __EXPOSURE_MAX
    global __EXPOSURE_MIN
	
    exposure_text = __GUI_DICT['exposure_text'].text
    if not exposure_text:
        return

    exposure = float(exposure_text)

    # Update slider to match text
    __GUI_DICT['exposure_slider'].eventson = False
    __GUI_DICT['exposure_slider'].set_val(exposure)
    __GUI_DICT['exposure_slider'].eventson = True

    # Set exposure for cameras
    exposure=min(__EXPOSURE_MAX,exposure)
    exposure=max(__EXPOSURE_MIN,exposure)
    spincam.set_exposure(exposure)
	
def __b_text(_=None):
    # Exposure text callback
    global __BVAL
	
    bval_text = __GUI_DICT['b_text'].text
    if not bval_text:
        return

    __BVAL = int(bval_text)

    set_bval(__BVAL)
	
def __g_text(_=None):
    # Exposure text callback
    global __GVAL
	
    gval_text = __GUI_DICT['g_text'].text
    if not gval_text:
        return

    __GVAL = int(gval_text)

    set_gval(__GVAL)
	
def set_gval(gval):	
	print("GVAL: "+str(gval))
	if(gval < 16):
		padg='0'
	else:
		padg=''
	ledserial.send(bytes.fromhex('0c '+padg+hex(gval).split('x')[1]))
	
def set_bval(bval):
	print("BVAL: "+str(bval))
	if(bval < 16):
		padb='0'
	else:
		padb=''
	ledserial.send(bytes.fromhex('0d '+padb+hex(bval).split('x')[1]))

def __ledr(_=None):
	ledserial.send(bytes.fromhex('0b64'))
	ledserial.send(bytes.fromhex('0c00'))
	ledserial.send(bytes.fromhex('0d00'))
	ledserial.send(bytes.fromhex('0e00'))
	
def __ledg(_=None):
	ledserial.send(bytes.fromhex('0b00'))
	ledserial.send(bytes.fromhex('0c64'))
	ledserial.send(bytes.fromhex('0d00'))
	ledserial.send(bytes.fromhex('0e00'))
	
def __ledb(_=None):
	ledserial.send(bytes.fromhex('0b00'))
	ledserial.send(bytes.fromhex('0c00'))
	ledserial.send(bytes.fromhex('0d64'))
	ledserial.send(bytes.fromhex('0e00'))
	
def __ledy(_=None):
	ledserial.send(bytes.fromhex('0b00'))
	ledserial.send(bytes.fromhex('0c00'))
	ledserial.send(bytes.fromhex('0d00'))
	ledserial.send(bytes.fromhex('0e64'))
	
def __noled(_=None):
	ledserial.send(bytes.fromhex('0b00'))
	ledserial.send(bytes.fromhex('0c00'))
	ledserial.send(bytes.fromhex('0d00'))
	ledserial.send(bytes.fromhex('0e00'))
	
def __ledbg(_=None):
	ledserial.send(bytes.fromhex('0b00'))
	ledserial.send(bytes.fromhex('0c64'))
	ledserial.send(bytes.fromhex('0d64'))
	ledserial.send(bytes.fromhex('0e00'))
	
def __mp_save(save_type):   
    global p
    global __GUI_DICT
    global __CAM
    global __IMSHOW_DICT
    global __HIST_DICT
    global __SAVE
    global __QUEUE
    global __STREAM

    __SAVE=1
    
    if not __STREAM:
        raise RuntimeError('Stream has not been started yet! Please start it first.')

    name_format = __GUI_DICT['name_format_text'].text
    num_images = int(__GUI_DICT['num_images_text'].text)
 
    num_to_avg = int(__GUI_DICT['avg_images_text'].text)
    directory = __GUI_DICT['directory_text'].text
    img_name = name_format.replace('{date}',str(datetime.date.today()))
	
    if directory:
	    directory = directory + '\\'
	
    img_main = directory + img_name.replace(' ', '_').replace('.', '_').replace(':', '')
    img_name = img_main + '.tiff'

    print('Experiment start: ' + str(datetime.datetime.now()))
    __STREAM = False
    #__save_images(img_name,num_to_avg,num_images, __CAM)
    spincam.__cleanup_cam()
    p = mp.Process(target=__save_images, args=(img_name,num_to_avg,num_images,__QUEUE))
    p.start()
    print(str(datetime.datetime.now()))
    
    
def __save_images(img_name, num_to_avg, num_images,__QUEUE):

    #pool = mp.Pool(mp.cpu_count())
    spincam.find_cam("stuff")
    spincam.init_cam()
    spincam.start_acquisition()
    for i in range(num_images):
        image_dict = spincam.get_image_and_avg(num_to_avg)
        if 'data' in image_dict:
            try:
                np = mp.Process(target=save_image, args=(img_name, image_dict))
                np.start()
                #tf.imsave(img_name, image_dict['data'].astype(np.uint16), compress=0, append=True)
                #pool.apply_async(save_image, args=(img_name, image_dict), callback=collect_result)
                print('Finished Acquiring ' + img_name)
                __QUEUE.put(image_dict['data'])
            except:
                print(sys.exc_info()[0])
                
    spincam.__cleanup_cam()
                
    #pool.close()
    print('Experiment end: ' + str(datetime.datetime.now()))
                
def collect_result(result):
    print('Saved')

def save_image(img_name, image_dict):
    tf.imsave(img_name, image_dict['data'].astype(np.uint16), compress=0, append=True)
	
def __spincam_gui():

    # Get figure
    fig = plt.figure()

    # Set position params
    padding = 0.01
    options_height = 0.02
    num_options = 6
    cam_plot_height_offset = num_options*options_height+num_options*padding
    cam_plot_width = 0.75
    cam_plot_height = 1-cam_plot_height_offset
	
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
	# Display Camera Image
    display_pos = [0,
                       cam_plot_height_offset,
                       cam_plot_width,
                       cam_plot_height]
    display_dict = __cam_plot(fig,
                                       display_pos,
                                       'Camera',
                                       options_height,
                                       padding)
    # Set initial values
    # Set callbacks
    display_dict['find_and_init_button'].on_clicked(__find_and_init_cam)
	
    stage_pos = [cam_plot_width,
                                    cam_plot_height_offset,
                                   0.25,
                                   cam_plot_height]
					
			
    # Creates stage controls
    but_width=options_height*2
    but_height=options_height*3
	

    red_but_pos = [display_pos[0]+display_pos[3]-4*padding,
                      display_pos[1]+display_pos[3]-2*padding-but_height,
                       but_width,
                       options_height*2]
    red_but_axes = fig.add_axes(red_but_pos)
    red_but = Button(red_but_axes, 'R')
    red_but.label.set_fontsize(7)
	
    yellow_but_pos = [red_but_pos[0]+padding+but_width,
                      red_but_pos[1],
                       but_width,
                       options_height*2]
    yellow_but_axes = fig.add_axes(yellow_but_pos)
    yellow_but = Button(yellow_but_axes, 'Y')
    yellow_but.label.set_fontsize(7)

    green_but_pos = [yellow_but_pos[0]+padding+ but_width,
                      yellow_but_pos[1],
                       but_width,
                       options_height*2]
    green_but_axes = fig.add_axes(green_but_pos)
    green_but = Button(green_but_axes, 'G')
    green_but.label.set_fontsize(7)
	
    blue_but_pos = [green_but_pos[0]+padding+ but_width,
                      green_but_pos[1],
                       but_width,
                       options_height*2]
    blue_but_axes = fig.add_axes(blue_but_pos)
    blue_but = Button(blue_but_axes, 'B')
    blue_but.label.set_fontsize(7)
	
    b_text_pos = [red_but_pos[0],
                      display_pos[3]-padding-but_height,
                       0.1,
                       options_height*2]
    b_text_axes = fig.add_axes(b_text_pos)
    b_text = TextBox(b_text_axes, 'bval')
    b_text.label.set_fontsize(7)
    b_text.set_val(1)
	
    g_text_pos = [red_but_pos[0],
                      display_pos[3]-10*padding-but_height,
                       0.1,
                       options_height*2]
    g_text_axes = fig.add_axes(g_text_pos)
    g_text = TextBox(g_text_axes, 'gval')
    g_text.label.set_fontsize(7)
    g_text.set_val(1)
	
   	
    red_but.on_clicked(__ledr)
    yellow_but.on_clicked(__ledy)
    blue_but.on_clicked(__ledb)
    green_but.on_clicked(__ledg)
    b_text.on_submit(__b_text)
    g_text.on_submit(__g_text)
									
	# Start stream
    start_stream_button_pos = [padding,
                               display_pos[1]-options_height,
                               0.5-2*padding,
                               options_height]
    start_stream_button_axes = fig.add_axes(start_stream_button_pos)
    start_stream_button = Button(start_stream_button_axes, 'Start Stream')
    start_stream_button.label.set_fontsize(7)
    
	# Set callback
    start_stream_button.on_clicked(__start_stream)

	# Stop stream
    stop_stream_button_pos = [start_stream_button_pos[0]+start_stream_button_pos[2]+2*padding,
                              display_pos[1]-options_height,
                              0.5-2*padding,
                              options_height]
    stop_stream_button_axes = fig.add_axes(stop_stream_button_pos)
    stop_stream_button = Button(stop_stream_button_axes, 'Stop Stream')
    stop_stream_button.label.set_fontsize(7)
    
	# Set callback
    stop_stream_button.on_clicked(__stop_stream)
	
	# fps
    fps_pos = [0, start_stream_button_pos[1]-options_height-padding, 1, options_height]
    (fps_slider, fps_text) = __slider_with_text(fig,
                                                                  fps_pos,
                                                                  'FPS',
                                                                  __FPS_MIN,
                                                                  __FPS_MAX,
                                                                  __FPS_MIN,
                                                                  padding)
    # Set callbacks
    fps_slider.on_changed(__fps_slider)
    fps_text.on_submit(__fps_text)

    # Exposure
    exposure_pos = [0, fps_pos[1]-options_height-padding, 1, options_height]
    (exposure_slider, exposure_text) = __slider_with_text(fig,
                                                          exposure_pos,
                                                          'Exposure',
                                                          __EXPOSURE_MIN,
                                                          __EXPOSURE_MAX,
                                                          __EXPOSURE_MIN,
                                                          padding)
    # Set callbacks
    exposure_slider.on_changed(__exposure_slider)
    exposure_text.on_submit(__exposure_text)

	#Set default directory to current
    directory_pos = [padding, exposure_pos[1]-options_height-padding, 0.75-2*padding, options_height]
    directory_axes = fig.add_axes(directory_pos)
    directory_text = TextBox(directory_axes, 'Directory')
    directory_text.label.set_fontsize(7)
    directory_text.set_val('')
	
    directory_button_pos = [directory_pos[0]+directory_pos[2]+padding,
                directory_pos[1],
                0.25-2*padding,
                options_height]
    directory_button_axes = fig.add_axes(directory_button_pos)
    directory_button = Button(directory_button_axes, 'Choose Directory')
    directory_button.label.set_fontsize(7)
    
    directory_button.on_clicked(__choose_directory)

    # Set name format
    name_format_pos = [(0.5-2*padding)*0.1875+2*padding,
                       directory_pos[1]-options_height-padding,
                       (0.5-2*padding)*0.8125-padding,
                       options_height]
    name_format_axes = fig.add_axes(name_format_pos)
    name_format_text = TextBox(name_format_axes, 'Name format')
    name_format_text.label.set_fontsize(7)
    name_format_text.set_val('test_{date}')
    
	    # Set save primary button
    save_button_pos = [padding,
                               name_format_pos[1]-options_height-padding,
                               (0.5-4*padding)/3,
                               options_height]
    save_button_axes = fig.add_axes(save_button_pos)
    save_button = Button(save_button_axes, 'Start Acquisition')
    save_button.label.set_fontsize(7)
    # Set callback
    save_button.on_clicked(__mp_save)
	
	# Set num images text
    num_images_text_pos = [cam_plot_width-20*padding,
                       padding,
                       0.1-2*padding,
                       options_height]
    num_images_text_axes = fig.add_axes(num_images_text_pos)
    num_images_text = TextBox(num_images_text_axes, '# to Acquire')
    num_images_text.label.set_fontsize(7)
    num_images_text.set_val(1)
	
	# Set num images text
    time_images_text_pos = [0.75,
                   directory_pos[1]-options_height-padding,
                   0.125,
                   options_height]
    time_images_text_axes = fig.add_axes(time_images_text_pos)
    time_images_text = TextBox(time_images_text_axes, 'Time between frames (s)')
    time_images_text.label.set_fontsize(7)
    time_images_text.set_val(0)
	
	# Set num images text
    avg_images_text_pos = [save_button_pos[0]+save_button_pos[2]+padding+(0.5-2*padding)*0.1875+2*padding,
                           name_format_pos[1]-options_height-padding,
                           (0.2-2*padding)*0.8125-padding,
                           options_height]
    avg_images_text_axes = fig.add_axes(avg_images_text_pos)
    avg_images_text = TextBox(avg_images_text_axes, '# to Avg')
    avg_images_text.label.set_fontsize(7)
    avg_images_text.set_val(10)
	
    # Set save primary button
    bulk_button_pos = [num_images_text_pos[0]+num_images_text_pos[2]+padding,
                           name_format_pos[1]-options_height-padding,
                           (0.2-2*padding)*0.8125-padding,
                           options_height]
    bulk_button_axes = fig.add_axes(bulk_button_pos)
    bulk_button = Button(bulk_button_axes, 'Zero Bulk')
    bulk_button.label.set_fontsize(7)
    # Set callback
    bulk_button.on_clicked(__bulk_save)
	
    return {'fig': fig,
            'display_dict': display_dict,
			'start_stream_button': start_stream_button,
            'stop_stream_button': stop_stream_button,
			'save_button': save_button,
			'name_format_text': name_format_text,
			'num_images_text': num_images_text,
			'avg_images_text': avg_images_text,
			'fps_slider': fps_slider,
			'fps_text': fps_text,
			'exposure_slider': exposure_slider,
			'exposure_text': exposure_text,
			'time_images_text': time_images_text,
			'directory_text': directory_text,
			'directory_button': directory_button,
			'bulk_button': bulk_button,
			'red_but': red_but,
			'yellow_but': yellow_but,
			'green_but': green_but,
			'blue_but': blue_but,
			'b_text': b_text,
			'g_text': g_text}

def __bulk_save(_=None):
    __save_images('bulk')
			
def __stream_images():

    global __IMSHOW_DICT 
    global __HIST_DICT
    global __QUEUE
    global __BITS
    global __STREAM

    try:

        if (__STREAM == 1):
            image_dict = spincam.get_image()
            __BITS = image_dict['bitsperpixel']
            __QUEUE.put(image_dict['data'])
        
    except:
        if __STREAM:
 
            raise
	
def main():
    global __QUEUE
    global __STREAM
    global __IMSHOW_DICT, __IMSHOW_SECONDARY_DICT
    global __HIST_DICT, __HIST_SECONDARY_DICT
    global __GUI_DICT
    global __BITS

    # Set GUI
    __GUI_DICT = __spincam_gui()

    # Update plot while figure exists
    while plt.fignum_exists(__GUI_DICT['fig'].number): # pylint: disable=unsubscriptable-object
        try:
            # Handle streams
            if __STREAM:
                __stream_images()

        # Make sure images are complete
            if not __QUEUE.empty():
            # Plot image and histogram
                __IMSHOW_DICT, __HIST_DICT = __plot_image_and_hist(__QUEUE.get(),
                                                                               2**__BITS-1,
                                                                               __GUI_DICT['display_dict']['image_axes'],
                                                                               __IMSHOW_DICT,
                                                                               __GUI_DICT['display_dict']['hist_axes'],
                                                                               __HIST_DICT)
            # Update plot
            plt.pause(sys.float_info.min)

            
        except: # pylint: disable=bare-except
            if plt.fignum_exists(__GUI_DICT['fig'].number):
                # Only re-raise error if figure is still open
                raise

    # Clean up
    __QUEUE = Queue()
    __STREAM = False
    __IMSHOW_DICT = {'imshow': None, 'imshow_size': None}
    __HIST_DICT = {'bar': None}
    __GUI_DICT = None

    print('Exiting...')

    return 0


if __name__ == '__main__':

    sys.exit(main())

		