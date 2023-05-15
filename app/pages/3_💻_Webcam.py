#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Created By   : Charley ∆. Lebarbier
# Date Created : Sunday 07 May. 2023
# ==============================================================================
# Script to detect Fire and/or Smoke into a webcam stream
# ==============================================================================

import av
import cv2 as cv
import numpy as np
import streamlit as st
import supervision as sv

from datetime import datetime
from pymongo import MongoClient
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO


## -- Do the fire/smoke detection
model = YOLO('../model/fire_best_v8.pt')




class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        """
        Read webcam stream and use the yolo custom model to detect
        fire and smoke
        """

        img = frame.to_ndarray(format="bgr24")

        box_annotator = sv.BoxAnnotator(
            thickness=1,
            text_thickness=1,
            text_scale=0.5
        )

        result = model(img, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        labels = [
            f"{model.model.names[class_id]} {round(100 * float(confidence),1)}%"
            for xy, _, confidence, class_id, _
            in detections
        ]

        final_frame = box_annotator.annotate(
            scene=img, 
            detections=detections, 
            labels=labels
        )

        labels = {0: u'__background__', 1: u'fire', 2: u'smoke'}

        for detect in detections:
            name_file = f"webcam : {datetime.now()}"
            name_label = labels[int(detect[3])+1]
            percent_conf = str(round(100 * float(detect[2]),1)) + "%"
            x_center, y_center = int(detect[0][0]), int(detect[0][1])
            width, height = int(detect[0][2]), int(detect[0][3])

            # print(name_file, img, img_width, img_height, name_label, 
            #            percent_conf, x_center, y_center, width, height)

            save_in_db(name_file, img, name_label, 
                    percent_conf, x_center, y_center, width, height)

        return av.VideoFrame.from_ndarray(final_frame, format="bgr24")


def save_in_db(name_file, frame, name_label, percent_conf, 
               x_center_bb, y_center_bb, width_bb, height_bb):
    """Connect to the MongoDB to save data"""

    ## -- Encoding the original image into binary (tobytes)
    byte_encode = frame.tobytes()

    data = {
        "Filename": name_file,
        "Frame Encode": byte_encode,
        # "Frame Size": {"Width": img_width, "Height": img_height},
        "Name Label": name_label,
        "Percent of Confidence": percent_conf,
        "Bounding Boxes (Yolo Format)": {"X Center": x_center_bb,
                                         "Y Center": y_center_bb,
                                         "Width": width_bb,
                                         "Height": height_bb}}

    try:
        client = MongoClient('mongodb://root:example@localhost:27017/')
        db = client['Fire_Detection']
        Collection = db['Webcam']
        Collection.insert_one(data)

    except Exception as e:
        raise e




################################################################################
############################### STREAMLIT APP ##################################
################################################################################

## -- Config
st.set_page_config(page_title='Fire Detection - Webcam', 
                   page_icon='💻', 
                   layout='wide')

## -- Title
st.title("💻 Webcam : : Detect 'Fire'🔥 - 'Smoke'🌫")

## -- Content
webrtc_streamer(
    key="detect_fire",
    rtc_configuration={
        "iceServers":
            [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
