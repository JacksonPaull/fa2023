import cv2
import matplotlib.pyplot as plt
import os

def crop(image, imshape):
    w, h = (image.shape[0]-imshape[0]) // 2, (image.shape[1] - imshape[1]) // 2
    return image[w:w+imshape[0], h:h+imshape[1], :]


def clean_dir(d):
    for f in os.listdir(d):
        if f.endswith('.gitkeep'):
            continue
        os.remove(os.path.join(d, f))

def extract_video(vid_filepath, dest_folder, n_frames=-1, show_first_frame=False, crop_size=(1080,1080)):
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

    while success:
        cv2.imwrite(f'{dest_folder}/frame{count:04d}.jpg', image)     # save frame as JPEG file      
        success, image = vidcap.read()
        if image is None:
            break

        image = crop(image, (1080, 1080))
        count += 1
        if (not n_frames == -1) and count > n_frames:
            break

    print('Successfully read and saved ', count, 'frames')
    vidcap.release()
    return count, fps
    

def encode_video(folder, output_fp='./output.mp4', fps=1):
    images = [img for img in os.listdir(folder) if img.endswith('.png')]
    frame = cv2.imread(os.path.join(folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_fp, fourcc, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(folder, image)))

    cv2.destroyAllWindows()
    video.release()