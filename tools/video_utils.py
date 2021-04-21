import cv2
import numpy
import os
from decord import VideoReader


def check_gray(chip):
    r, g, b = cv2.split(chip)
    r = r.astype(numpy.float32)
    g = g.astype(numpy.float32)
    b = b.astype(numpy.float32)
    s_w, s_h = r.shape[:2]
    x = (r + b + g) / 3

    area_s = s_w * s_h
    # x = chip_gray
    r_gray = abs(r - x)
    g_gray = abs(g - x)
    b_gray = abs(b - x)
    r_sum = numpy.sum(r_gray) / area_s
    g_sum = numpy.sum(g_gray) / area_s
    b_sum = numpy.sum(b_gray) / area_s
    gray_degree = (r_sum + g_sum + b_sum) / 3
    if gray_degree < 3:
        return False, gray_degree
    else:
        return True, gray_degree


def if_video_active(frame_list):
    pre_frame = None  # 总是取视频流前一帧做为背景相对下一帧进行比较
    diff = 10
    threshold = 50
    for num in range(0, len(frame_list), 2):
        frame_lwp_cv = frame_list[num]

        # 判断灰度图，并修改阈值
        if num == 0:
            result, degree = check_gray(frame_lwp_cv)
            if result:
                diff = 25
                threshold = 50

        h, w, _ = frame_lwp_cv.shape
        frame_lwp_cv = frame_lwp_cv[int(h * 0.107):int(h * 0.921), int(0.0568 * w):]
        try:
            # 加异常捕获的原因是，某些帧是空帧，所以转不成，做特殊处理
            gray_lwp_cv = cv2.cvtColor(frame_lwp_cv, cv2.COLOR_BGR2GRAY)  # 转灰度图
        except Exception:
            continue

        gray_lwp_cv = cv2.resize(gray_lwp_cv, (224, 224))
        # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。
        # 对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
        gray_lwp_cv = cv2.GaussianBlur(gray_lwp_cv, (21, 21), 0)
        # 在完成对帧的灰度转换和平滑后，就可计算与背景帧的差异，并得到一个差分图（different map）。
        # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
        if pre_frame is None:
            pre_frame = gray_lwp_cv
        else:
            img_delta = cv2.absdiff(pre_frame, gray_lwp_cv)
            # 由于我们的图片本身就有点灰色的效果，所以把两张图的色差降低
            thresh = cv2.threshold(img_delta, diff, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < threshold:  # 设置敏感度
                    continue
                else:
                    return True
            pre_frame = gray_lwp_cv
    return False


def video_legal(video_path, remove_illegal=True):
    # 检查切出来的视频是否可读
    try:
        cv_wrong = False
        cap = cv2.VideoCapture(str(video_path))
        flag, f = cap.read()
        if not flag:
            cv_wrong = True
        cap.release()
        if cv_wrong:
            # opencv检测出视频不正常，删掉
            print("opencv remove", video_path, flush=True)
            if remove_illegal:
                os.remove(str(video_path))
            return False

        vr = VideoReader(str(video_path))
        last_timestamp = [-100, -100]
        video_wrong = False if (len(vr) > 1 and len(vr.get_key_indices()) > 1) else True
        if video_wrong:
            print("len vr {} and  len(vr.get_key_indices()) {}".format(len(vr), len(vr.get_key_indices())),
                  video_path, flush=True)
            if remove_illegal:
                os.remove(str(video_path))
            return False

        for i in range(len(vr)):
            # 解决卡住
            tmp = vr.get_frame_timestamp(i)
            if last_timestamp[0] == tmp[0] and last_timestamp[1] <= tmp[1]:
                print("last_timestamp", i, video_path, last_timestamp[0], tmp[0], last_timestamp[1], tmp[1],
                      flush=True)
                video_wrong = True
                break
            last_timestamp = tmp
        if video_wrong:
            print("timestamp error", video_path, flush=True)
            if remove_illegal:
                os.remove(str(video_path))
            return False
        for i in range(len(vr)):
            # 解决decord._ffi.base.DECORDError: [18:09:55] /io/decord/src/video/ffmpeg -loglevel quiet/threaded_decoder.cc:288: [18:08:29] /io/decord/src/video/ffmpeg -loglevel quiet/threaded_decoder.cc:216: Check failed: avcodec_send_packet(dec_ctx_.get(), pkt.get()) >= 0 (-1094995529 vs. 0) Thread worker: Error sending packet.
            try:
                _t = vr[i]
            except Exception as e:
                print("DECORDError", e, i, video_path, flush=True)
                if remove_illegal:
                    os.remove(str(video_path))
                return False
    except:
        print("remove video format exception", video_path, flush=True)
        if remove_illegal:
            os.remove(str(video_path))
        return False
    return True


if __name__ == '__main__':
    from pathlib import Path
    from mmcv import Config
    from tqdm import tqdm
    from mmaction.datasets.pipelines import Compose
    from railway.utils import utils
    import shutil
    import sys
    import time

    infer_pipeline_config = Config.fromfile('infer_pipeline.py')
    base_pipeline = Compose(infer_pipeline_config.base_decode_pipeline)
    hand_watch_pipleline = Compose(infer_pipeline_config.hand_watch_pipleline)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    video_paths = Path(input_dir).glob('**/*.mp4')
    utils.mkdir(output_dir)
    all_video_count = 0
    still_count = 0
    check_times = []
    for video_path in tqdm(list(video_paths)):
        all_video_count += 1
        video_path = str(video_path)
        _d = dict(
            filename=video_path,
            label=-1,
            start_index=0,
            modality='RGB')
        _base_data = base_pipeline(_d)
        img_data = _base_data['imgs']
        t1 = time.time()
        is_active = if_video_active(img_data)
        check_times.append(time.time() - t1)
        if not is_active:
            still_count += 1
            shutil.copy(video_path, output_dir)

    print(input_dir, all_video_count, still_count, sum(check_times) / len(check_times))
