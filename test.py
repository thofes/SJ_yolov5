
import subprocess

subprocess.call(["python3", "detect_big.py", "--weights", "/home/thomas/yolov5/weights_skijumper_only.pt", "--name", "test", "--save-crop", "--source", "/media/thomas/Hofes/Bilder_BA/runnos/hd4_runno/354", "--project", "/home/thomas/test/", "--save-txt"])

