# Anomaly-Detection

### Directory Structure:

Anomaly-Detection
- app
  - static
    - anoframes
    - css
      - main.css
    - vids
  - templates
    - includes
      - formhelpers.html
    - base.html
    - index.html
    - login.html
    - main.html
    - register.html
    - video.html
  - app.py
  - forms.py
  - framesHandler.py
  - vidProcDbModels.py
  - vidProcWatchdog.py
- models
  - alexnet2.pth
  - anogan.h5
  - discriminator_model.h5
  - obj.names
  - yolov4-obj_last_2.weights
  - yolov4-obj.cfg
- testdata
  - fire1.mp4
- final.py
- vidlogs.db


## Execution

### Video Processing:

1. Open 4 command prompts, and go to your working directory. (activate virtual environment if applicable)
2. Starting the processes:
  - CMD1: ```python app\app.py``` for starting flask application
  - CMD2: ```python app\vidProcWatchdog.py``` for starting Watchdog
  - CMD3: ```python app\framesHandler.py``` for automatic video creation and alert generation
3. Copy testing video in testdata folder and give it's relative path in final.py line 247
4. Run final.py (CMD4: ```python final.py```)

### App:

1. Open web app with the address shown in CMD1 (http://127.0.0.1:5000/)
2. Register and login
3. Click on _List of videos_ to view logs
4. Click on a log (blue hyperlink) to view generated video

### Alternate Execution Method :
1. final_year_project\Scripts\activate
2. run.bat
