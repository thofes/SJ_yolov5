import os
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess

xx = '11'


landing_pic = "/media/thomas/Hofes/BilderEwoxxMCI/ÖMBhofen/Competition1/" + xx


Path_Result = '/home/thomas/Yolo/show/' + xx
if os.path.isdir(Path_Result):
  print("Path exists")
else:
  os.mkdir(Path_Result)
  print("Path was not existing")


name = 'detections' 


#Landing and Flight
subprocess.call(["python3", "detect_Auswertung.py", "--weights", "/home/thomas/yolov5/best_landing_flight.pt", "--name", name, "--save-crop", "--source", landing_pic, "--project", Path_Result, "--save-txt", "--max-det", "1", "--conf-thres", "0.8"])


Path = Path_Result


with open(Path + '/' + name + '/list.json', "r") as fp:
    data = json.load(fp)
  
conf = np.array(data["conf"])
index = np.array(data["index"])
cls = np.array(data["cls"])
x_center = np.array(data["x"])
y_center = np.array(data["y"])

"""
if int(xx) < 10: 
    with open(Path[:-1] + "results.json", "r") as fp:
        resultdata = json.load(fp)
else:
    with open(Path[:-2] + "results.json", "r") as fp:
        resultdata = json.load(fp)
"""

#resultdata["Ordner"].append(xx)
#resultdata["SOLL"].append(Labels[xx])

X = index
Y = conf

plt.plot(X, Y)
#plt.show()

index1 = []
conf1 = []
cls1 = []
index2 = []
conf2 = []
cls2 = []
index3 = []
conf3 = []
cls3 = []
index4 = []
conf4 = []
cls4 = []


#Für Ramsau
for i, x in enumerate(index):
    if x[0] == '1':
      conf1.append(conf[i])
      index1.append(x[2:])
      cls1.append(cls[i])
    elif x[0] == '2':
      conf2.append(conf[i])
      index2.append(x[2:])
      cls2.append(cls[i])
    elif x[0] == '3':
      conf3.append(conf[i])
      index3.append(x[2:])
      cls3.append(cls[i])
    elif x[0] == '4':
      conf4.append(conf[i])
      index4.append(x[2:])
      cls4.append(cls[i])
"""
#Für BHofen
for i, x in enumerate(index):
    if x[0] == '1':
      conf2.append(conf[i])
      index2.append(x[2:])
      cls2.append(cls[i])
    elif x[0] == '2':
      conf1.append(conf[i])
      index1.append(x[2:])
      cls1.append(cls[i])
    elif x[0] == '3':
      conf4.append(conf[i])
      index4.append(x[2:])
      cls4.append(cls[i])
    elif x[0] == '4':
      conf3.append(conf[i])
      index3.append(x[2:])
      cls3.append(cls[i])
"""



plt.plot(index2, conf2)
#plt.show()
plt.plot(index1, conf1)
#plt.show()
plt.plot(index4, conf4)
#plt.show()
plt.plot(index3, conf3)
#plt.show()

plt.plot(index1, conf1)
plt.plot(index2, conf2)
plt.plot(index3, conf3)
plt.plot(index4, conf4)
#plt.show()






 
cams = [cls1, cls2, cls3, cls4]
indexes = [index1, index2, index3, index4]
detections = []
for idx, cam in enumerate(cams):
    lastcls = 9
    for i, c in enumerate(cam):
        if c == 1:
            lastcls = 1
        if c == 0 and lastcls == 1:
            detections.append(str(idx+1)+ "_" + indexes[idx][i])
            lastcls = 9
            break
    if len(detections) > 0:
        break



print("The Preselection would detect landings in following frames: ", detections)
    #resultdata["Selection"].append(detections)



#print("By manually labeling following frame is the landing: ", Labels[xx])




#with open(Path[:-2] + "results.json", "w") as fp:
    #json.dump(resultdata, fp, indent=4)

index = data["index"].index(str(detections[0]))


dataset = {
      "path":[],
      "x":[],
      "y":[]
    }   
path_to_crop = Path + '/' + name + '/crops/Cam' + str(detections[0]) + '.jpg'
dataset["path"].append(path_to_crop)  
dataset["x"].append(data["x"][index])
dataset["y"].append(data["y"][index])

with open(Path + '/' + name + "/res.json", "w") as fp:
    json.dump(dataset, fp, indent=4)
