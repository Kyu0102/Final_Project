import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import cv2 as cv
import numpy as np
import urllib.request
import os

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
filename = 'imagenet_classes.txt'
urllib.request.urlretrieve(url, filename)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained = True)
model.to(device)
model.eval()

cap = cv.VideoCapture("test_video.mp4")

frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

area_size = (frameWidth + frameHeight)/2

fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=0)

while(cap.isOpened()):
    _, frame = cap.read()

    fgmask = fgbg.apply(frame)

    _, _, stats, centroids = cv.connectedComponentsWithStats(fgmask)

    for index, centroid in enumerate(centroids):
        if (stats[index][0] == 0 and stats[index][1] == 0) or np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > area_size:
            cv.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))
            count += 1
            if count % 4 == 0:
                crop_img = frame[y:y+height, x:x+width]
                image = Image.fromarray(crop_img)
                img_transformed = transform(image)
                input = img_transformed.unsqueeze(0)
                output = model(input.to(device))
            
                score, predicted = torch.max(output.data, 1)
            
                with open('imagenet_classes.txt') as f:
                    classes = [line.strip() for line in f.readlines()]

                class_label = classes[predicted.item()]
                
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, class_label, (x,y), font, 0.5, (0,255,0), 1, cv.LINE_AA)

    cv.imshow('frame',frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
