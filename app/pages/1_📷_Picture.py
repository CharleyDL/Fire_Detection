#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Created By   : Charley ∆. Lebarbier
# Date Created : Thursday 13 Apr. 2023
# ==============================================================================
# Script to detect Fire and/or Smoke into a image JPG/JPEG
# ==============================================================================

import cv2 as cv
import numpy as np
import streamlit as st
import supervision as sv

from PIL import Image
from pymongo import MongoClient
from ultralytics import YOLO


## -- Do the fire/smoke detection
model = YOLO('../model/fire_best_v8.pt')


def save_in_db(name_file, img, img_width, img_height, name_label, percent_conf, 
               x_center_bb, y_center_bb, width_bb, height_bb):
    """Connect to the MongoDB to save detection data"""

    ## -- Encoding the original image into binary (tobytes)
    img_encode = cv.imencode('.jpg', img)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()

    data = {
        "Filename": name_file,
        "Image Encode": byte_encode,
        "Image Size": {"Width": img_width, "Height": img_height},
        "Name Label": name_label,
        "Percent of Confidence": percent_conf,
        "Bounding Boxes (Yolo Format)": {"X Center": x_center_bb,
                                         "Y Center": y_center_bb,
                                         "Width": width_bb,
                                         "Height": height_bb}
    }

    try:
        client = MongoClient('mongodb://root:example@localhost:27017/')
        db = client['Fire_Detection']
        Collection = db['Image']
        Collection.insert_one(data)

    except Exception as e:
        raise e


def fire_detect(upload):
    """Use to detect fire and smoke in image using YOLO with transfer learning

    Parameters
    ----------
    upload: required
        Image format (jpg, jpeg, png) where to apply the detection
    """

    ## -- Get Image, check the size and resize it
    original_image = Image.open(upload)
    img_width, img_height = original_image.size
    max_size = 640

    img_np = np.asarray(original_image)
    if img_width > max_size or img_height > max_size:
        scale = max_size / max(img_width, img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        img_np = cv.resize(img_np, (new_width, new_height))

    ## -- Detect the Fire
    img = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)

    result = model(img, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)

    labels = [f"{model.model.names[class_id]} {round(100 * float(confidence),1)}%"
                for xy, _, confidence, class_id, _ 
                in detections]

    ## -- Customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5
    )

    ## -- Annotate the img with BBox
    final_img = box_annotator.annotate(
        scene=img, 
        detections=detections, 
        labels=labels
    )

    final_img = cv.cvtColor(final_img, cv.COLOR_RGB2BGR)

    ## -- Display Original and Result Img
    col1, col2 = st.columns(2)

    col1.write("Original Image 📷")
    col1.image(original_image)

    col2.write("Detection Result 🔍")
    col2.image(final_img, clamp=True, channels='RGB')

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

        save_in_db(name_file, img, img_width, img_height, name_label, 
                   percent_conf, x_center, y_center, width, height)




################################################################################
############################### STREAMLIT APP ##################################
################################################################################

## -- Config
st.set_page_config(page_title='Fire Detection - Picture', 
                   page_icon='📷', 
                   layout='wide')

## -- Title
st.title("📷 Picture : Detect 'Fire'🔥 - 'Smoke'🌫")

## -- Content
pic_upload = st.file_uploader('', type=["jpg", "jpeg", "png"])

if pic_upload is not None:
    fire_detect(upload=pic_upload)
else:
    st.subheader("No image uploaded yet")