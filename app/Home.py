#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Created By   : Charley ∆. Lebarbier
# Date Created : Thursday 13 Apr. 2023
# ==============================================================================


import streamlit as st


# Config
st.set_page_config(page_title='Fire Detection', page_icon=':fire:', layout='wide')

# Title
st.title('Detect Fire and Smoke with YoloV8x')

# Content
st.write(
    """
    <b>The application presented here is designed to detect fires (fire and smoke) 
    from static or dynamic images, then save data into a MongoDB</b>  

    Currently, it supports three types of format: photos📷, video🎥 
    and live webcam💻.
    """, unsafe_allow_html=True
)

st.markdown('----')

c1, c2 = st.columns(2)
with c1:
    st.subheader('Methodology')
    st.write(
        """
        After a manual labeling of images, mentioning fire and smoke, we have 
        chosen to use the <b><i>pre-trained YOLO V8x model</i></b> to apply 
        transfer learning by passing it our custom dataset.  

        <i><b>A second pre-trained model be used with YOLO V5x to compare 
        result. Nevertheless, in this application, only YOLO V8x 
        is used with our custom weights.</b></i>
        """, unsafe_allow_html=True
    )

with c2:
    st.subheader('Features in Progress')
    st.write(
        """
        ----- Page to watch the saving data  
        ----- Video detection from an url  
        ----- Saving the detection on your computer (Image, Video, Webcam), not only in DB
        """
    )

st.markdown('----')

st.subheader('Future Works')
c1, c2 = st.columns(2)
with c1:
    st.write(
        """
        To be notified about our future projects, you can follow me !  
        Feel free to share your feedback, suggestions, and also critics with me.
        """
    )
with c2: 
    st.info('**[@Charley∆.L.](https://github.com/CharleyDL)**', 
            icon="💻")

