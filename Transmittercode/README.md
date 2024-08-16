import cv2
import numpy as np
import requests
from datetime import datetime

# ThingSpeak API endpoint URL
api_url = 'https://api.thingspeak.com/update'
api_key = 'PV6CRIWEKLYVEBGL'  # Replace with your Write API Key

# Load YOLO model and its configuration for YOLOv3
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO dataset names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Initialize the webcam with the correct camera index
cap = cv2.VideoCapture(1)  # Change the index to 0 if this doesn't work

person_detected = False
start_time = None

while True:
    ret, frame = cap.read()

    # Perform blob from the frame and forward pass through YOLO network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []
    Width = frame.shape[1]
    Height = frame.shape[0]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class 0 is for 'person'
                center_x, center_y, w, h = (detection[0:4] * np.array([Width, Height, Width, Height])).astype(int)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            person_detected = True
            start_time = datetime.now()
            break

    if person_detected:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        power_consumption = elapsed_time / 3600  # in watt-hours

        print("Person detected")
        print(f"Power Consumption: {power_consumption:.2f} Wh")
        payload = {'api_key': api_key, 'field1': 1, 'field2': power_consumption}
    else:
        print("No person detected")
        payload = {'api_key': api_key, 'field1': 0, 'field2': 0}

    try:
        # Make a POST request to the ThingSpeak API
        response = requests.post(api_url, params=payload)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("ThingSpeak Update Successful")
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        # Handle any exceptions that may occur during the request
        print("Error:", str(e))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture object
cap.release()
cv2.destroyAllWindows()
