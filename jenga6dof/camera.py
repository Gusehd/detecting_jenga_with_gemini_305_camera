"""Orbbec Gemini 305 RGB-D wrapper (pyorbbecsdk2): BGR image + depth aligned to color."""
import cv2
import numpy as np
from pyorbbecsdk import (AlignFilter, Config, OBFormat, OBSensorType, OBStreamType,
                         Pipeline)


def _color_to_bgr(frame):
    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt in (OBFormat.YUYV, OBFormat.YUY2):
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    raise RuntimeError(f"unsupported color format {fmt}")


DEFAULT_PRESET = "Close Range High Accuracy"
DEFAULT_DEPTH_RES = (1280, 800)


class OrbbecRGBD:
    def __init__(self, width=1280, height=720, fps=30, depth_res=DEFAULT_DEPTH_RES,
                 preset=DEFAULT_PRESET):
        self.pipeline = Pipeline()
        dev = self.pipeline.get_device()
        if preset:
            try:
                dev.load_preset(preset)
            except Exception as e:
                print(f"[camera] preset '{preset}' not applied: {e}")
        try:
            self.preset = dev.get_current_preset_name()
        except Exception:
            self.preset = "?"
        config = Config()
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            cp = color_profiles.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        except Exception:
            cp = color_profiles.get_default_video_stream_profile()
        config.enable_stream(cp)
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            dp = depth_profiles.get_video_stream_profile(depth_res[0], depth_res[1],
                                                         OBFormat.Y16, fps)
        except Exception:
            dp = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(dp)
        dv = dp.as_video_stream_profile()
        self.depth_size = (dv.get_width(), dv.get_height())
        self.config = config
        self.color_profile = cp.as_video_stream_profile()
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def start(self):
        self.pipeline.start(self.config)
        intr = self.color_profile.get_intrinsic()
        dist = self.color_profile.get_distortion()
        self.K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])
        # BROWN_CONRADY_K6 -> OpenCV rational model order (k1,k2,p1,p2,k3,k4,k5,k6)
        self.dist = np.array([dist.k1, dist.k2, dist.p1, dist.p2,
                              dist.k3, dist.k4, dist.k5, dist.k6])
        self.size = (intr.width, intr.height)
        return self

    def read(self, timeout_ms=200):
        """Returns (bgr uint8 HxWx3, depth float32 HxW in meters, 0 = invalid) or None."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None
        frames = self.align.process(frames)
        if frames is None:
            return None
        frames = frames.as_frame_set()
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if color is None or depth is None:
            return None
        bgr = _color_to_bgr(color)
        dw, dh = depth.get_width(), depth.get_height()
        d = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(dh, dw)
        depth_m = d.astype(np.float32) * depth.get_depth_scale() / 1000.0
        return bgr, depth_m

    def stop(self):
        self.pipeline.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
