# PyHoloAssist

The hand_eye_project.py file projects hand poses to images after using expoerter app in the PSI repo (https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter).

To use this code, please download the HoloAssist dataset (https://holoassist.github.io/) and the *cam_info.tar* file on the website. You need to untar into the same folder.


## How to install?
The code is tested with Python==3.8. Required packages are listed in the ```requirements.txt``` file.


## How to run the script?

```
hand_eye_projection.py  --eye --folder_path "folder location" video_name "sequence name" --save_eyeproj --save_video
```
If you want to save projected eye gaze, using save_eyeproj flag without save_video will increase the processing speed. The output, "Eye_proj.txt" will be located in the Eye folder in the same sequence folder.
