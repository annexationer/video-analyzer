import cv2
import numpy as np
import os


def calc_histogram(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_frame[:, :, 0]
    hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def detect_scene_change(prev_hist, curr_hist):
    # Use correlation as the method of comparison
    comparison = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    # A lower threshold will result in more scene changes being detected, but may also result in more false positives. Conversely, a higher threshold will result in fewer scene changes being detected, but may also result in more false negatives. You may want to start with a threshold of 0.8 and adjust it as necessary based on the results you get.
    threshold = 0.95  # set a threshold for the comparison result
    if comparison < threshold:
        return True
    else:
        return False


def detect_intro_outro(video_path: str | os.PathLike, flag: bool, filepath: str) -> tuple[int, int, int, str] | None:
    filename = os.path.splitext(os.path.basename(video_path))[0]
    video_path = os.fspath(video_path)
    video_capture = cv2.VideoCapture(video_path)

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
    prev_frame = None
    prev_hist = None
    curr_frame: np.ndarray

    for frame_index in range(50, frame_count):
        ret, curr_frame = video_capture.read()

        if not ret:
            break

        if frame_index / video_fps > 60:
            break

        curr_hist = calc_histogram(curr_frame)

        if prev_hist is not None and detect_scene_change(prev_hist, curr_hist):
            intro_end_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            intro_frame = curr_frame.copy()
            if flag:
                cv2.imwrite(os.path.join(filepath, f'{filename}-{frame_index}-intro.jpeg'), intro_frame)
                cv2.imwrite(os.path.join(filepath, f'{filename}-{frame_index - 1}-intro.jpeg'), prev_frame)
            break

        prev_hist = curr_hist
        prev_frame = curr_frame.copy()

    if video_duration >= 180:  # duration >= 3m
        outro_frame_start_index = int((video_duration - 90) * video_fps)
    else:
        outro_frame_start_index = frame_count // 2

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, outro_frame_start_index)

    prev_frame = None
    prev_hist = None

    while True:
        ret, curr_frame = video_capture.read()
        outro_frame_start_index += 1
        if not ret:
            break

        curr_hist = calc_histogram(curr_frame)

        if prev_hist is not None and detect_scene_change(prev_hist, curr_hist):
            outro_start_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            outro_frame = curr_frame.copy()
            if flag:
                cv2.imwrite(os.path.join(filepath, f'{filename}-{outro_frame_start_index}-outro.jpeg'), outro_frame)
                cv2.imwrite(os.path.join(filepath, f'{filename}-{outro_frame_start_index - 1}-outro.jpeg'), prev_frame)
            break

        prev_hist = curr_hist
        prev_frame = curr_frame.copy()

    if intro_end_time is None:
        intro_end_time = 0.0
    if outro_start_time is None:
        outro_start_time = video_duration

    video_capture.release()

    return round(intro_end_time), round(outro_start_time), round(video_duration), video_definition


print(detect_intro_outro('/Users/equalizer/Video/探究吧！第一季03.ts', True, '/Users/equalizer/Video'))
