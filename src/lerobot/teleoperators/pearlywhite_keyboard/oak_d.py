import depthai
import cv2
import numpy as np

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection
)

import torch
from typing import List

class OakDProDevice():
    def __init__(self, mxid=None):

        model_file_path_ = f"/home/mailroom/lerobot/src/lerobot/teleoperators/keyboard/model_5_rtdetr"
        self.processor = AutoImageProcessor.from_pretrained(
            model_file_path_,
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_file_path_,
        ).to("cpu")
        self.custom_threshold = 0.35

        self.frame = None
        self.disparity = None
        self.depth = None
        self.mxid = mxid

        self.depth_width = 640
        self.depth_height = 360
        self.rgb_width = 640*4
        self.rgb_height = 360*4

        self.pipeline = depthai.Pipeline()
        self.cam_rgb = self.pipeline.createColorCamera()
        self.cam_rgb.setPreviewSize(self.rgb_width, self.rgb_height)

        self.cam_rgb.initialControl.setManualFocus(115)
        self.cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)
        self.cam_rgb.initialControl.setManualExposure(20000, 600)
        self.cam_rgb.initialControl.setSharpness(1)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
        self.xout_rgb = self.pipeline.createXLinkOut()
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)
        self.cam_rgb.initialControl.setManualExposure(20000, 750)
        self.cam_rgb.initialControl.setSharpness(4)

        self.od_calibration = ((357, 330), (1839, 1094))

        # Define sources and outputs
        self.monoLeft = self.pipeline.create(depthai.node.MonoCamera)
        self.monoRight = self.pipeline.create(depthai.node.MonoCamera)
        self.stereo = self.pipeline.create(depthai.node.StereoDepth)

        # Properties
        self.monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoLeft.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        self.monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoRight.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

        self.stereo.initialConfig.setConfidenceThreshold(255)
        self.stereo.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        self.stereo.setDepthAlign(camera = depthai.CameraBoardSocket.RGB)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setOutputSize(self.monoLeft.getResolutionWidth(), self.monoLeft.getResolutionHeight())
        self.stereo.setSubpixel(False)
        self.stereo.setExtendedDisparity(True)

        # Linking
        self.monoLeft.out.link(self.stereo.left)
        self.monoRight.out.link(self.stereo.right)

        xoutDepth = self.pipeline.create(depthai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        self.stereo.depth.link(xoutDepth.input)

        xoutDepth = self.pipeline.create(depthai.node.XLinkOut)
        xoutDepth.setStreamName("disp")
        self.stereo.disparity.link(xoutDepth.input)
        
        self.device = depthai.Device(self.pipeline, depthai.DeviceInfo(mxid))
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)

        self.depthQueue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.dispQ = self.device.getOutputQueue(name="disp", maxSize=4, blocking=False)

    def oak_device(self):
        return self.device
    
    def oak_config(self):
        return self.depth_width, self.depth_height, self.rgb_width, self.rgb_height

    def run(self):
        in_rgb = self.q_rgb.tryGet()

        if in_rgb is not None:
            self.frame = in_rgb.getCvFrame()
            self.od_rgb = self.frame[self.od_calibration[0][1]:self.od_calibration[1][1], 
                                  self.od_calibration[0][0]:self.od_calibration[1][0]]
            cv2.imshow('frame', self.od_rgb)
            results = self.rtdetr_detections([self.od_rgb])
            print(results)
        cv2.waitKey(1)

        in_depthData = self.depthQueue.get()
        in_disparityData = self.dispQ.get() #.getFrame()

        if in_depthData is not None:
            self.depth = in_depthData

        if in_disparityData is not None:
            self.disparity = in_disparityData.getFrame()

        if self.disparity is not None:
            self.disparity = (self.disparity * (255 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
            self.disparity = cv2.applyColorMap(self.disparity, cv2.COLORMAP_JET)
        
        return self.frame, self.disparity, in_depthData
    
     # Detect with Real Time Model
    def rtdetr_detections(self, frames: List):
        inputs = self.processor(images=frames, return_tensors="pt").to("cpu")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [(frame.shape[0], frame.shape[1]) for frame in frames]

        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.custom_threshold)

        return results
        
        
def main(args=None):
    oak = OakDProDevice('18443010D1AEAC0F00')
    while True:
         oak.run()

    
if __name__ == "__main__":
    main()
    
