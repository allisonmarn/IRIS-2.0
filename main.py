import multiprocessing as mp
import numpy as np
from time import time
import multiprocessing as mp
import spincam


def save_image(img_name, image_dict):
    ski.imsave(img_name, image_dict['data'].astype(np.uint16), compress=0, append=True)
    return 0

def collect_result(result):
    print("callback")
    
def main():
    counter=0
    num_images = 100
    num_to_avg = 100
    for i in range(num_images):
        image_dict = spincam.get_image_and_avg(num_to_avg)
        # Make sure images are complete
        if 'data' in image_dict:
        # Save image
            try:
                pool.apply_async(save_image, args=(img_name, image_dict), callback=collect_result)
                print('Finished Acquiring ' + img_name)
                img_name= img_main + '_' + str(file_number) + '.tiff'
                file_number = file_number + 1
            except:
                print("Error")

    pool.close()
    pool.join()  
    
    return 0

    

if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    results = []
    main()