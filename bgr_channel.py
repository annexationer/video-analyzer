import sys, os, cv2, traceback

import numpy as np


def detect_intro_outro(video_path: str | os.PathLike, flag: bool, filepath: str) -> tuple[int, int, int, str] | None:
    """
    Your current approach of detecting intro and outro times based on the mean pixel value of grayscale frames has a few limitations. It assumes that the intro is darker (mean pixel value < 10) and the outro is brighter (mean pixel value > 100). This might not always be the case for all videos.
    Here are a few suggestions to improve the success rate:
    Use more features: Instead of just using the mean pixel value, you can use more features of the video frames. For example, you can use the standard deviation of pixel values, the color histogram, etc. This will give you a more comprehensive understanding of the video content.
    Use machine learning: If you have a labeled dataset of videos with known intro and outro times, you can train a machine learning model to predict the intro and outro times based on the features of the video frames. This will likely give you a higher success rate than a heuristic approach.
    Use audio features: Often, the intro and outro of a video have distinctive audio characteristics, such as theme music. You can extract audio features from the video and use them to detect the intro and outro times.
    Use scene change detection: Intros and outros often coincide with scene changes. You can use scene change detection algorithms to detect the intro and outro times.
    Improve the granularity of your search: Instead of breaking after the first 3 minutes, you can continue scanning the video but with a less frequent sampling rate. This might help to catch intros and outros that are longer than usual.
    """
    filename = os.path.splitext(os.path.basename(video_path))[0]
    video_path = os.fspath(video_path)
    video_capture = cv2.VideoCapture(video_path)

    threshold_upper_limit = 230

    if not video_capture.isOpened():
        return

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_duration = frame_count / video_fps

    if frame_width >= 1280 and frame_height >= 710:
        video_definition = '高清'
    else:
        video_definition = '标清'

    intro_end_time = None
    outro_start_time = None
    intro_frame = None
    outro_frame = None
    frame: np.ndarray

    for frame_index in range(50, frame_count):
        ret, frame = video_capture.read()

        if not ret:
            break

        if frame_index / video_fps > 60:  # Restrict to first 1 minutes
            break

        # calculates the mean value of an image or a specific region of interest (ROI)
        # src: This is the source image or the image from which you want to calculate the mean value.
        # mask (optional): This is an optional mask image that specifies the region of interest. If provided, the mean value will be calculated only for the pixels within the mask. If not provided, the mean value will be calculated for the entire image.
        # The cv2.mean function returns a tuple containing the mean values for each channel (B, G, R, and optionally alpha) of the image or ROI.
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_mean = cv2.mean(grayscale_frame)
        grayscale_mean_pixel_value = grayscale_mean[0]

        mean = cv2.mean(frame)
        mean_pixel_value_b = mean[0]  # Blue channel
        mean_pixel_value_g = mean[1]  # Green channel
        mean_pixel_value_r = mean[2]  # Red channel

        if (grayscale_mean_pixel_value < 10 or
            # grayscale_mean_pixel_value > threshold_upper_limit or
            mean_pixel_value_b > threshold_upper_limit or  # Check if blue channel exceeds threshold_upper_limit
            mean_pixel_value_g > threshold_upper_limit or  # Check if green channel exceeds threshold_upper_limit
            mean_pixel_value_r > threshold_upper_limit  # Check if red channel exceeds threshold_upper_limit
        ) and intro_end_time is None:
            print('intro', frame, grayscale_mean, mean)
            # intro_end_time = frame_index / video_fps
            intro_end_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            intro_frame = frame.copy()
            break

    if video_duration >= 180:  # duration >= 3m
        outro_frame_start_index = int((video_duration - 90) * video_fps)
    else:
        outro_frame_start_index = frame_count // 2

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, outro_frame_start_index)

    while True:
        ret, frame = video_capture.read()
        outro_frame_start_index += 1
        if not ret:
            break

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_mean = cv2.mean(grayscale_frame)
        grayscale_mean_pixel_value = grayscale_mean[0]

        mean = cv2.mean(frame)
        mean_pixel_value_b = mean[0]  # Blue channel
        mean_pixel_value_g = mean[1]  # Green channel
        mean_pixel_value_r = mean[2]  # Red channel

        if (grayscale_mean_pixel_value < 10 or
            # grayscale_mean_pixel_value > threshold_upper_limit or
            mean_pixel_value_b > threshold_upper_limit or  # Check if blue channel exceeds threshold_upper_limit
            mean_pixel_value_g > threshold_upper_limit or  # Check if green channel exceeds threshold_upper_limit
            mean_pixel_value_r > threshold_upper_limit  # Check if red channel exceeds threshold_upper_limit
        ) and outro_start_time is None:
            print('outro', frame, grayscale_mean, mean)
            # outro_start_time = (outro_frame_start_index + 1) / video_fps
            outro_start_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            outro_frame = frame.copy()
            break

    # for frame_index in range(frame_count - 1, -1, -1):
    #     video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    #     ret, frame = video_capture.read()
    #
    #     if not ret:
    #         break
    #
    #     if video_duration - (frame_index / video_capture.get(cv2.CAP_PROP_FPS)) > 180:  # Restrict to last 3 minutes
    #         break
    #
    #     grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     mean = cv2.mean(grayscale_frame)
    #     mean_pixel_value = mean[0]
    #     if mean_pixel_value > 100 and outro_start_time is None:
    #         outro_start_time = (frame_index + 1) / video_capture.get(cv2.CAP_PROP_FPS)
    #         break

    # Additional logic
    if flag:
        if intro_frame is not None:
            cv2.imwrite(os.path.join(filepath, f'{filename}-intro.jpeg'), intro_frame)
        if outro_frame is not None:
            cv2.imwrite(os.path.join(filepath, f'{filename}-outro.jpeg'), outro_frame)

    if intro_end_time is None:
        intro_end_time = 0.0
    if outro_start_time is None:
        outro_start_time = video_duration

    video_capture.release()

    return round(intro_end_time), round(outro_start_time), round(video_duration), video_definition


print(detect_intro_outro('/Users/equalizer/Video/13.ts', True, '/Users/equalizer/Video'))
