from pathlib import Path
import os

import cv2
import subprocess
from tools import video_utils
import sys
import time
import shutil


def get_one_frame(ffmpeg_cmd_template, output_frame_path=None):
    store = True
    if output_frame_path is None:
        store = False
        output_frame_path = '/tmp/' + ''.join(random.sample(string.ascii_letters + string.digits, 20)) + '.jpg'
    cmd = ffmpeg_cmd_template.format(output_frame_path)
    subprocess.call(cmd, shell=True)
    frame = cv2.imread(output_frame_path)
    if not store:
        subprocess.call('rm {}'.format(output_frame_path), shell=True)
    return frame


def get_first_frame(video_path, output_frame_path=None):
    template = 'ffmpeg -loglevel quiet -y -ss 0 -i ' + video_path + '  -vframes 1 {}'
    return get_one_frame(template, output_frame_path=output_frame_path)


def slice_one_video_custom_interval(v_path: Path, _do_keep_clips, do_first_frames, need_remove_illegal,
                                    output_path: Path, interval=20):
    t1 = time.time()
    print("%.2f is running task %s" % (t1, str(v_path)), flush=True)
    tmp_path = output_path / v_path.name.replace('.mp4', '_tmp.mp4')

    clear_audio_cmd = 'ffmpeg -loglevel quiet -y -i {} -vcodec copy -an {}'.format(str(v_path).replace(' ', '\ '),
                                                                                   str(tmp_path))
    subprocess.call(clear_audio_cmd, shell=True)

    t2 = time.time()
    print("%.2f clear_audio_cmd task %s %.2f " % (time.time(), str(v_path), t2 - t1), flush=True)

    t1 = time.time()
    segment_list_path = output_path / (v_path.name[: -4] + '.csv')
    stream_clip_cmd = 'ffmpeg -loglevel quiet -y -i {} -map 0:v -c copy -f segment -segment_time {} -reset_timestamps 1 -segment_list_type csv -segment_list {} {}-%5d.mp4'.format(
        str(tmp_path), str(interval), str(segment_list_path), str(output_path / v_path.name[: -4])
    )
    subprocess.call(stream_clip_cmd, shell=True)

    output_paths = []
    for path in output_path.glob("{}-*.mp4".format(v_path.name[:-4])):
        output_paths.append(path)

    t2 = time.time()
    print("%.2f clip video task %s %.2f" % (time.time(), str(v_path), t2 - t1), flush=True)
    t1 = time.time()

    if do_first_frames:
        for video_path in output_paths:
            frame_path = str(video_path).replace('.mp4', '_first.jpg')
            get_first_frame(str(video_path), frame_path)
    t2 = time.time()
    print("%.2f first frames task %s %.2f " % (time.time(), str(v_path), t2 - t1), flush=True)

    t1 = time.time()
    if _do_keep_clips:
        if need_remove_illegal:
            for video_path in output_paths:
                video_utils.video_legal(str(video_path))
    else:
        for video_path in output_paths:
            os.remove(str(video_path))
    t2 = time.time()
    print("%.2f check video legal %s %.2f " % (time.time(), str(v_path), t2 - t1), flush=True)

    if tmp_path.exists():
        os.remove(str(tmp_path))


if __name__ == "__main__":
    _base_video_path = sys.argv[1]
    _do_keep_clips = sys.argv[2] == 'True'
    _do_first_frames = sys.argv[3] == 'True'
    _need_remove_illegal = sys.argv[4] == 'True'
    _output_dir_path = sys.argv[5]
    interval = 2
    if len(sys.argv) > 6:
        interval = int(sys.argv[6])

    slice_one_video_custom_interval(Path(_base_video_path), _do_keep_clips, _do_first_frames, _need_remove_illegal,
                                    Path(_output_dir_path), interval)
