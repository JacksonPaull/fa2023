import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(image, imshape):
    w, h = (image.shape[0]-imshape[0]) // 2, (image.shape[1] - imshape[1]) // 2
    return image[w:w+imshape[0], h:h+imshape[1], :]


def extract_video(vid_filepath, dest_folder, n_frames=-1, show_first_frame=False, crop_size=(1080,1080)):
    vidcap = cv2.VideoCapture(vid_filepath)
    success, image = vidcap.read()
    count = 0

    # First frame
    image = crop(image, crop_size)
    if show_first_frame:
        plt.imshow(image[:,:,::-1])
        plt.axis('off')
        plt.title('Frame 0')
        plt.show()

    while success:
        cv2.imwrite(f'{dest_folder}/frame{count}.jpg', image)     # save frame as JPEG file      
        success, image = vidcap.read()
        if image is None:
            break

        image = crop(image, (1080, 1080))
        count += 1
        if (not n_frames == -1) and count > n_frames:
            break

    print('Successfully read and saved ', count, 'frames')
    vidcap.release()
    return count
    