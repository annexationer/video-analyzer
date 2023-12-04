import sys, os, cv2, traceback
import pandas as pd
from pathlib import Path


def detect_intro_outro(video_path: str | os.PathLike) -> tuple[int, int, int, str] | None:
    """
    18 254
    Your current approach of detecting intro and outro times based on the mean pixel value of grayscale frames has a few limitations. It assumes that the intro is darker (mean pixel value < 10) and the outro is brighter (mean pixel value > 200). This might not always be the case for all videos.
    Here are a few suggestions to improve the success rate:
    Use more features: Instead of just using the mean pixel value, you can use more features of the video frames. For example, you can use the standard deviation of pixel values, the color histogram, etc. This will give you a more comprehensive understanding of the video content.
    Use machine learning: If you have a labeled dataset of videos with known intro and outro times, you can train a machine learning model to predict the intro and outro times based on the features of the video frames. This will likely give you a higher success rate than a heuristic approach.
    Use audio features: Often, the intro and outro of a video have distinctive audio characteristics, such as theme music. You can extract audio features from the video and use them to detect the intro and outro times.
    Use scene change detection: Intros and outros often coincide with scene changes. You can use scene change detection algorithms to detect the intro and outro times.
    Improve the granularity of your search: Instead of breaking after the first 3 minutes, you can continue scanning the video but with a less frequent sampling rate. This might help to catch intros and outros that are longer than usual.

    """
    video_path = os.fspath(video_path)

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        return

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = frame_count / video_capture.get(cv2.CAP_PROP_FPS)

    if frame_width >= 1280 and frame_height >= 720:
        video_definition = '高清'
    else:
        video_definition = '标清'

    intro_end_time = None
    outro_start_time = None

    for frame_index in range(frame_count):
        ret, frame = video_capture.read()

        if not ret:
            break

        if frame_index / video_capture.get(cv2.CAP_PROP_FPS) > 180:  # Restrict to first 3 minutes
            break

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculates the mean value of an image or a specific region of interest (ROI)
        # src: This is the source image or the image from which you want to calculate the mean value.
        # mask (optional): This is an optional mask image that specifies the region of interest. If provided, the mean value will be calculated only for the pixels within the mask. If not provided, the mean value will be calculated for the entire image.
        # The cv2.mean function returns a tuple containing the mean values for each channel (B, G, R, and optionally alpha) of the image or ROI.
        mean = cv2.mean(grayscale_frame)
        mean_pixel_value = mean[0]

        if mean_pixel_value < 10 and intro_end_time is None:
            intro_end_time = frame_index / video_capture.get(cv2.CAP_PROP_FPS)

    for frame_index in range(frame_count - 1, -1, -1):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_capture.read()

        if not ret:
            break

        if video_duration - (frame_index / video_capture.get(cv2.CAP_PROP_FPS)) > 180:  # Restrict to last 3 minutes
            break

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = cv2.mean(grayscale_frame)
        mean_pixel_value = mean[0]
        if mean_pixel_value > 200 and outro_start_time is None:
            outro_start_time = (frame_index + 1) / video_capture.get(cv2.CAP_PROP_FPS)
            break

    # Additional logic
    if intro_end_time is None:
        intro_end_time = 0.0
    if outro_start_time is None:
        outro_start_time = video_duration

    video_capture.release()

    return round(intro_end_time), round(outro_start_time), round(video_duration), video_definition


def convert_seconds_to_hms(total_seconds: int):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    else:
        return f'{minutes:02d}:{seconds:02d}'


def main(args: list[str] = None):
    compatible = True
    if sys.version_info < (3, 11):
        compatible = False
    elif not hasattr(sys, 'base_prefix'):
        compatible = False
    if not compatible:
        raise ValueError('This script is only for use with Python >= 3.11')
    else:
        import argparse

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-d', '--dir', metavar='directory', action='store', dest='directory', required=True, type=Path, help='directory containing videos or video file to analysis')

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        usage=None,
        prefix_chars='-',
        add_help=True,
        epilog='epilog',
        allow_abbrev=True,
        exit_on_error=True,
        parents=[parent_parser],
        description='analysis video and output result to xlsx'
    )

    # parser.add_argument('filename', default='.', type=argparse.FileType(mode='w', encoding='utf-8', errors='strict'), help='destionation file')
    # parser.add_argument('-c', '--count', metavar='metavar', type=int, const=99, nargs='?')
    # parser.add_argument('-e', '--epilog', nargs='?', required=False, metavar='epilog')
    parser.add_argument('output', action='store', nargs='?', default=os.path.join(os.path.expanduser('~'), 'Desktop'), type=Path, help='directory to output result')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s 1.0.0')
    namespace = parser.parse_args(args, namespace=argparse.Namespace(foo=99))
    filepath: Path = namespace.directory.expanduser()
    output: Path = namespace.output.expanduser()
    print(filepath, output)
    if not filepath.exists():
        raise ValueError(f'directory {filepath} not exists')

    if str(output) != '-' and not output.exists():
        raise ValueError(f'output {output} not exists')

    # Record = namedtuple('Record', '文件名称,文件格式,片头结束时间,片尾开始时间,片头时长,片尾时长,时长_秒,时长_分钟,运行时间,清晰度,文件大小(KiB),文件大小(MiB)')

    records = {'文件名称': [], '文件格式': [], '片头结束时间': [], '片尾开始时间': [], '片头时长': [], '片尾时长': [], '时长(秒)': [], '时长(分钟)': [], '运行时间': [], '清晰度': [], '文件大小(KiB)': [], '文件大小(MiB)': []}

    if filepath.is_dir():
        for filename in filepath.iterdir():
            if filename.is_file():
                v = detect_intro_outro(filename)
                if v:
                    filesize = os.path.getsize(filename)
                    kilobytes = round(filesize / 1024, 2)
                    megabytes = round(kilobytes / 1024, 2)
                    # record = Record(filename.stem, filename.suffix.replace('.', ''), v[0], v[1], v[0], v[2] - v[1], v[2], v[2] // 60, convert_seconds_to_hms(v[2]), v[3], kilobytes, megabytes)
                    # records.append(record)
                    records['文件名称'].append(filename.stem)
                    records['文件格式'].append(filename.suffix.replace('.', ''))
                    records['片头结束时间'].append(v[0])
                    records['片尾开始时间'].append(v[1])
                    records['片头时长'].append(v[0])
                    records['片尾时长'].append(v[2] - v[1])
                    records['时长(秒)'].append(v[2])
                    records['时长(分钟)'].append(round(v[2] / 60, 2))
                    records['运行时间'].append(convert_seconds_to_hms(v[2]))
                    records['清晰度'].append(v[3])
                    records['文件大小(KiB)'].append(kilobytes)
                    records['文件大小(MiB)'].append(megabytes)

    elif filepath.is_file():
        v = detect_intro_outro(filepath)
        if v:
            filesize = os.path.getsize(filepath)
            kilobytes = round(filesize / 1024, 2)
            megabytes = round(kilobytes / 1024, 2)
            # record = Record(filepath.stem, filepath.suffix.replace('.', ''), v[0], v[1], v[0], v[2] - v[1], v[2], v[2] // 60, convert_seconds_to_hms(v[2]), v[3], kilobytes, megabytes)
            # records.append(record)
            records['文件名称'].append(filepath.stem)
            records['文件格式'].append(filepath.suffix.replace('.', ''))
            records['片头结束时间'].append(v[0])
            records['片尾开始时间'].append(v[1])
            records['片头时长'].append(v[0])
            records['片尾时长'].append(v[2] - v[1])
            records['时长(秒)'].append(v[2])
            records['时长(分钟)'].append(round(v[2] / 60, 2))
            records['运行时间'].append(convert_seconds_to_hms(v[2]))
            records['清晰度'].append(v[3])
            records['文件大小(KiB)'].append(kilobytes)
            records['文件大小(MiB)'].append(megabytes)

    else:
        raise ValueError(f'Unknown type {type(filepath)}')

    df = pd.DataFrame(records)
    if str(output) == '-':
        print(df)
    else:
        df.to_excel(str(output / 'result.xlsx'), sheet_name=filepath.name, index=False)


if __name__ == '__main__':
    rc = 1
    try:
        main()
        rc = 0
    except Exception as e:
        traceback.print_exc()
        print('Error: %s' % e, file=sys.stderr)
    sys.exit(rc)
