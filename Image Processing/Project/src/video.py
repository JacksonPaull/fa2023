import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def crop(image, imshape):
    """
        Crop an image to a center rectangle with shape imshape

        Parameters:
            - image: np.array; A 2d Image 
    """
    w, h = (image.shape[0]-imshape[0]) // 2, (image.shape[1] - imshape[1]) // 2
    return image[w:w+imshape[0], h:h+imshape[1], :]


def clean_dir(d):
    """Remove all files except gitkeeps from a directory"""
    for f in os.listdir(d):
        if f.endswith('.gitkeep'):
            continue
        os.remove(os.path.join(d, f))

def extract_video(vid_filepath, dest_folder, n_frames=-1, show_first_frame=False, crop_size=(1080,1080)):
    """
        Extract all frames from a video and save as jpg's

        Parameters:
            vid_filepath: str
                The filepath to the video that will be extracted

            dest_folder: str
                The filepath to the directory to store all captured frames

            n_frames: int, default= -1
                The number of frames to store.
                If n_frames == -1, this parameter is ignored

            show_first_frame: bool, default = False
                Whether to display the first frame that is captured (useful for debugging)

            crop_size: tuple (int, int), default = (1080, 1080)
                Size of a center crop to apply to all images extracted
    """
    vidcap = cv2.VideoCapture(vid_filepath)
    fps = vidcap.get(cv2.CAP_PROP_FPS) 
    success, image = vidcap.read()
    count = 0

    # First frame
    image = crop(image, crop_size)
    if show_first_frame:
        plt.imshow(image[:,:,::-1])
        plt.axis('off')
        plt.title('Frame 0')
        plt.show()

    

    if n_frames == -1:
        while success:
            cv2.imwrite(f'{dest_folder}/frame{count:04d}.jpg', image)     # save frame as JPEG file      
            success, image = vidcap.read()
            if image is None or not success:
                break

            image = crop(image, (1080, 1080))
            count += 1


    else:
        for count in tqdm(range(n_frames)):
            cv2.imwrite(f'{dest_folder}/frame{count:04d}.jpg', image)     # save frame as JPEG file      
            success, image = vidcap.read()
            if image is None or not success:
                break

            image = crop(image, (1080, 1080))


    print('Successfully read and saved ', count, 'frames')
    vidcap.release()
    return count, fps
    

def encode_video(folder, output_fp='./output.mp4', fps=1):
    """
        Create a video from frames in a folder, with names that sort alphabetically to their order in the video

        Parameters:
            folder: str
                The folder from which to source all images for video creation
            
            output_fp: str
                The destination filepath of the output video
            
            fps: int or float
                The framerate of the output video in frames per second
    """
    images = [img for img in os.listdir(folder) if img.endswith('.png')]
    frame = cv2.imread(os.path.join(folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_fp, fourcc, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(folder, image)))

    video.release()