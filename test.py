import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

frames = pipeline.wait_for_frames()
color = frames.get_color_frame()

print("Frame OK:", color.get_width(), color.get_height())

pipeline.stop()
