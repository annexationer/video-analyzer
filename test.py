import sys

import cv2, time, os, warnings
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from typing import Sequence

warnings.filterwarnings('ignore', module='cv2')

os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"


def extract_common_part(str1: str, str2: str) -> str:
    common = ''
    for c1, c2 in zip(str1, str2, strict=False):
        if c1 == c2:
            common += 'c1'
        else:
            break
    return common


def compare_frames(frame1, frame2):
    # convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)


config = {
    '少儿': {
        'intro_correlation_coefficient': 0.56,
        'outro_correlation_coefficient': 0.68,
        'intro_duration_mean': 26,
        'outro_duration_mean': 17.8,
        'total_duration_mean': 438,
        'total_duration_std': 359.8
    }
}


def predict_duration(duration: float, coefficient=1.7, genre: str = '少儿'):
    v = config[genre]
    intro_duration = v['intro_duration_mean'] + v['intro_correlation_coefficient'] * (duration - v['total_duration_mean']) / v['total_duration_std']
    outro_duration = v['outro_duration_mean'] + v['outro_correlation_coefficient'] * (duration - v['total_duration_mean']) / v['total_duration_std']
    return min(intro_duration * coefficient, duration * 0.5), min(outro_duration * coefficient, duration * 0.5)


def detect_intro_outro(video_paths: Sequence[str | os.PathLike], threshold=0.85, frame_output_flag: bool = False, frame_output_path: str = '') -> tuple[int, int] | None:
    filename = extract_common_part(*(os.fspath(v) for v in video_paths))
    frame_output_path = os.path.dirname(video_paths[0])

    cap1 = cv2.VideoCapture(os.fspath(video_paths[0]))
    cap2 = cv2.VideoCapture(os.fspath(video_paths[1]))

    if not cap1.isOpened() or not cap2.isOpened():
        return

    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    min_frame_count = min(frame_count1, frame_count2)
    video_fps1 = cap1.get(cv2.CAP_PROP_FPS)
    video_fps2 = cap2.get(cv2.CAP_PROP_FPS)
    video_duration1: float = frame_count1 / video_fps1
    video_duration2: float = frame_count2 / video_fps2

    # if int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)) >= 1280 and int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)) >= 720:
    #     video_definition = '高清'
    # else:
    #     video_definition = '标清'

    # while True:
    #     ret1, frame1 = cap1.read()
    #     ret2, frame2 = cap2.read()
    #
    #     if not ret1 or not ret2:
    #         break
    #
    #     structural_similarity = compare_frames(frame1, frame2)
    #     if structural_similarity < threshold:
    #         print("Intro ends at frame:", frame_num)
    #         break
    #     # print(frame_num)
    #     frame_num += 1

    threshold_seconds = 180
    prev_frame1: np.ndarray | None = None
    prev_frame2: np.ndarray | None = None
    next_frame1: np.ndarray | None = None
    next_frame2: np.ndarray | None = None
    intro_duration = None
    outro_duration = None
    intro_duration_threshold = int(min(min_frame_count / 2, int(video_fps1 * threshold_seconds)))
    outro_duration_threshold1 = frame_count1 - int(min(frame_count1 / 2, int(video_fps1 * threshold_seconds)))
    outro_duration_threshold2 = frame_count2 - int(min(frame_count2 / 2, int(video_fps2 * threshold_seconds)))

    for frame_index in range(intro_duration_threshold):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        structural_similarity = compare_frames(frame1, frame2)

        if structural_similarity < threshold:
            print(f'intro>> {structural_similarity=:.5f}, frame_index: {frame_index:<20} CAP_PROP_POS_FRAMES: {cap1.get(cv2.CAP_PROP_POS_FRAMES), cap2.get(cv2.CAP_PROP_POS_FRAMES)}')
            if frame_output_flag:
                cv2.imwrite(os.path.join(frame_output_path, f'{os.path.basename(video_paths[0])}-intro-{frame_index}.jpeg'), frame1)
                cv2.imwrite(os.path.join(frame_output_path, f'{os.path.basename(video_paths[1])}-intro-{frame_index}.jpeg'), frame2)
                if prev_frame1 is not None:
                    cv2.imwrite(os.path.join(frame_output_path, f'{os.path.basename(video_paths[0])}-intro-prev-{frame_index - 1}.jpeg'), prev_frame1)
                if prev_frame2 is not None:
                    cv2.imwrite(os.path.join(frame_output_path, f'{os.path.basename(video_paths[1])}-intro-prev-{frame_index - 1}.jpeg'), prev_frame2)
            intro_duration = cap1.get(cv2.CAP_PROP_POS_MSEC) / 1000
            break

        prev_frame1 = frame1
        prev_frame2 = frame2

    if intro_duration is None:
        intro_duration = 0

    frame_index1 = frame_count1
    frame_index2 = frame_count2

    while True:
        frame_index1 -= 1
        frame_index2 -= 1

        if frame_index1 < outro_duration_threshold1 or frame_index2 < outro_duration_threshold2:
            break

        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_index1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_index2)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            continue

        structural_similarity = compare_frames(frame1, frame2)

        if structural_similarity < threshold:
            print(f'outro>> {structural_similarity=:.5f}, frame_index: {str((frame_index1, frame_index2)):<20} CAP_PROP_POS_FRAMES {cap1.get(cv2.CAP_PROP_POS_FRAMES), cap2.get(cv2.CAP_PROP_POS_FRAMES)}')
            if frame_output_flag:
                cv2.imwrite(os.path.join(frame_output_path, f'{os.path.basename(video_paths[0])}-outro-{frame_index1}.jpeg'), frame1)
                cv2.imwrite(os.path.join(frame_output_path, f'{os.path.basename(video_paths[1])}-outro-{frame_index2}.jpeg'), frame2)
                if next_frame1 is not None:
                    cv2.imwrite(os.path.join(frame_output_path, f'{os.path.basename(video_paths[0])}-outro-next-{frame_index1 + 1}.jpeg'), next_frame1)
                if next_frame2 is not None:
                    cv2.imwrite(os.path.join(frame_output_path, f'{os.path.basename(video_paths[1])}-outro-next-{frame_index2 + 1}.jpeg'), next_frame2)
            outro_duration = video_duration1 - cap1.get(cv2.CAP_PROP_POS_MSEC) / 1000
            # print(f'{video_duration1 - cap1.get(cv2.CAP_PROP_POS_MSEC) / 1000}, {video_duration2 - cap2.get(cv2.CAP_PROP_POS_MSEC) / 1000}')
            break

        next_frame1 = frame1
        next_frame2 = frame2

    if outro_duration is None:
        outro_duration = 0

    cap1.release()
    cap2.release()

    return intro_duration, outro_duration


# video_paths = ('/Users/equalizer/Video/猫狗勇者大联盟/猫狗勇者大联盟_16.ts', '/Users/equalizer/Video/猫狗勇者大联盟/猫狗勇者大联盟_19.ts')
# video_paths = ('/Users/equalizer/Video/test/奥特怪兽拟人化计划 怪兽娘 第一季_09/奥特怪兽拟人化计划 怪兽娘 第一季_05.ts', '/Users/equalizer/Video/test/奥特怪兽拟人化计划 怪兽娘 第一季_09/奥特怪兽拟人化计划 怪兽娘 第一季_09.ts')
# video_paths = ('/Users/equalizer/Video/test/爱探险的朵拉第三季/爱探险的朵拉第三季_07.ts', '/Users/equalizer/Video/test/爱探险的朵拉第三季/爱探险的朵拉第三季_08.ts')
# video_paths = ('/Users/equalizer/Video/test/爱探险的朵拉第六季/爱探险的朵拉第六季_04.ts', '/Users/equalizer/Video/test/爱探险的朵拉第六季/爱探险的朵拉第六季_05.ts')
# video_paths = ('/Users/equalizer/Video/test/爱探险的朵拉（第八季）/爱探险的朵拉（第八季）01.ts', '/Users/equalizer/Video/test/爱探险的朵拉（第八季）/爱探险的朵拉（第八季）20.ts')
# video_paths = ('/Users/equalizer/Video/test/艾米咕噜1（有号）/艾米咕噜1（有号）_06.ts', '/Users/equalizer/Video/test/艾米咕噜1（有号）/艾米咕噜1（有号）_10.ts')
# video_paths = ('/Users/equalizer/Video/test/艾米爱地球小课堂/艾米爱地球小课堂_05.ts', '/Users/equalizer/Video/test/艾米爱地球小课堂/艾米爱地球小课堂_08.ts')
video_paths = ('/Users/equalizer/Video/test/艾米咕噜涂鸦小课堂/艾米咕噜涂鸦小课堂_05.ts', '/Users/equalizer/Video/test/艾米咕噜涂鸦小课堂/艾米咕噜涂鸦小课堂_08.ts')
start_time = time.time()
print(detect_intro_outro(video_paths=video_paths, threshold=0.86, frame_output_flag=True, frame_output_path=''))
print('runtime', time.time() - start_time)
