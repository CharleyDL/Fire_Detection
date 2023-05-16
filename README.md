# About
<div align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" />
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" />
  <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" />
  
  </br>
  </br> 
  <p><i>This Application streamlit detect flames and smoke into 📷Picture, 🎥Video and 💻Webcam using Transfer Learning based on Yolov8x.
        Each image/frame (encoded into binary) and its detection (bounded boxes, labels, confidence...) are saved into a mongodb
  </i></p>
</div> 

# Install 🏗
1. Clone the repository where you want 
2. Install requirements.txt ``pip install -r requirements.txt``
3. Create a MongoDB call ``Fire_Detection`` with 3 collections : ``Image``, ``Video`` and ``Webcam``
4. Launch a terminal and navigate to the folder cloned *(normally call 'fire_detection')* 🫠  
  4.1 Navigate to the folder app  
  4.2 Launch the streamlit application with the command ``streamlit run Home.py``
5. Enjoy the experience
6. Consult your MongoDB to get the track of your detection

# Screenshot 🖼
<div style="display: flex;" align="center">
  <figure style="margin-right: 20px;">
    <img width="450" alt="Main Page" src="https://github.com/CharleyDL/fire_detection/assets/21961841/0169773f-3468-4c96-805b-3430909a63b4">
  </figure>
  
  <figure>
    <img width="450" alt="Picture Detection" src="https://github.com/CharleyDL/fire_detection/assets/21961841/0992918f-4749-48bf-9e79-9043cc1f6cb3">
  </figure>
  
  <figure>
    <img width="450" alt="Picture Detection" src="https://github.com/CharleyDL/fire_detection/assets/21961841/ad44b85f-51d6-4fe9-aa0e-7b197e5b1335">
  </figure>
  
  <figure>
    <img width="450" alt="Picture Detection" src="https://github.com/CharleyDL/fire_detection/assets/21961841/dca6d4c8-4198-4ee7-8e01-9f120f2dcde9">
  </figure>
</div>

# Next Features 🔮
- Upgrade the saving in DB (currently image-frame multiple detection -> one entry per detection | Future : image-frame with multiple detection -> one entry (all labels / bounding boxes)
- Page to watch the saving data
- Video detection from an url 
- Saving the detection on your computer (Button download), not only in DB
- Possibility to change weight / model
- Add a confidence cursor
- Add a loader during processing



# Cleanup for your own usage

You can safely remove the following files/folders from repo root:
- `.github/`
- `LICENSE`
- `README.md`
- `CHANGELOG.md`
