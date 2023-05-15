#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Created By   : Charley ∆. Lebarbier
# Date Created : Sunday 07 May. 2023
# ==============================================================================

import cv2 as cv
import imageio.v3 as iio
import numpy as np
import tempfile
import streamlit as st
import supervision as sv

from pymongo import MongoClient
from ultralytics import YOLO


## -- Do the fire/smoke detection
model = YOLO('../model/fire_best_v8.pt')




def save_in_db(name_file, frame, img_width, img_height, name_label, percent_conf, 
               x_center_bb, y_center_bb, width_bb, height_bb):
    """Connect to the MongoDB to save data"""

    ## -- Encoding the original image into binary (tobytes)
    img_encode = cv.imencode('.jpg', frame)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()

    data = {
        "Filename": name_file,
        "Frame Encode": byte_encode,
        "Frame Size": {"Width": img_width, "Height": img_height},
        "Name Label": name_label,
        "Percent of Confidence": percent_conf,
        "Bounding Boxes (Yolo Format)": {"X Center": x_center_bb, 
                                         "Y Center": y_center_bb, 
                                         "Width": width_bb,
                                         "Height": height_bb}}

    try:
        client = MongoClient('mongodb://root:example@localhost:27017/')
        db = client['Fire_Detection']
        Collection = db['Video']
        Collection.insert_one(data)

    except Exception as e:
        raise e


def fire_detect(upload):
    """Use to detect fire and smoke in video using YOLO with transfer learning

    Parameters
    ----------
    upload: required
        Video where to apply the detection
    """

    ## -- Create a temporary file to read the original video
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(upload.read())

    ## -- Get information about the video
    vid = cv.VideoCapture(tfile.name)
    vid_width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

    ## -- Create a file to Save the final video
    output_file = tempfile.NamedTemporaryFile(delete=False, 
                                              suffix=".mp4")
    writer = cv.VideoWriter(output_file.name, 
                            cv.VideoWriter_fourcc(*'DIVX'), 
                            7, 
                            (vid_width, vid_height))

    ## -- Customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5
    )

    while True:
        ret, frame = vid.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        ## -- Proceed the detection on each frame
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        labels = [f"{model.model.names[class_id]} {round(100 * float(confidence),1)}%" 
                  for xy, _, confidence, class_id, _ 
                  in detections]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        writer.write(frame)     # writing each frame to create the new video

        ## -- Prepare the necessary data to save them in DB
        labels = {0: u'__background__', 1: u'fire', 2: u'smoke'}

        for detect in detections:
            name_file = upload.name
            name_label = labels[int(detect[3])+1]
            percent_conf = str(round(100 * float(detect[2]),1)) + "%"
            x_center, y_center = int(detect[0][0]), int(detect[0][1])
            width, height = int(detect[0][2]), int(detect[0][3])

            # print(name_file, img, img_width, img_height, name_label, 
            #            percent_conf, x_center, y_center, width, height)

            save_in_db(name_file, frame, vid_width, vid_height, name_label, 
                    percent_conf, x_center, y_center, width, height)


    vid.release()
    writer.release()
    cv.destroyAllWindows()

    ## -- Convert the video in WebM (for streamlit)
    output_webm_file = tempfile.NamedTemporaryFile(delete=False, 
                                                   suffix=".webm")
    input_video = iio.imread(output_file.name)
    iio.imwrite(output_webm_file.name, input_video, fps=30, 
                codec="libvpx-vp9")

    st.video(data=output_webm_file.name)



################################################################################
############################### STREAMLIT APP ##################################
################################################################################

## -- Config
st.set_page_config(page_title='Fire Detection - Video', 
                   page_icon='🎥', 
                   layout='wide')

## -- Title
st.title("🎥 Video : Detect 'Fire'🔥 - 'Smoke'🌫")

## -- Content
## ---- Choose between URL or Video File
tab1, tab2 = st.tabs(["Upload File", "URL"])
with tab1:
    vid_upload = st.file_uploader('', type=["mp4", "avi", "mov", "mpv"])

    if vid_upload is not None:
        fire_detect(upload=vid_upload)
    else:
        st.subheader("No video uploaded yet")

with tab2:
    url = st.text_input('url', placeholder='http://url_video_.com')

    st.write("Not available yet, implementation in progress!")
    if st.button('Analyze') and url is not None:
        # fire_detect(upload=url)
        st.write("Not available yet, implementation in progress!")
