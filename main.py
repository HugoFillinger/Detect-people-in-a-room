import random
import ssl

import cv2
import numpy as np
from flask import Flask, jsonify, request
import base64

import mqtt
from yolo_segmentation import YOLOSegmentation

app = Flask(__name__)


# Définition d'une route pour l'API
@app.route('/api/chillCode', methods=['POST'])
def get_number_of_people_in_a_room():
    try:
        # Initialisation du compteur de personnes
        number_of_people = 0

        # Récupération et décodage de l'image encodée en base64
        base64_string = request.json['image']
        roomId = request.json['roomid']
        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        # Initialisation du détecteur YOLO
        yolo_segmentation = YOLOSegmentation("yolov8m-seg.pt")

        # Détection d'objets dans l'image
        bounding_boxes, classes, segmentations, scores = yolo_segmentation.detect(image)

        # Comptage du nombre de personnes détectées
        for _, class_id, _, _ in zip(bounding_boxes, classes, segmentations, scores):
            if class_id == 0:  # Assumer que l'ID 0 correspond à la classe 'personne'
                number_of_people += 1

        # Envoi du nombre de personnes au serveur MQTT
        mqtt.send_to_mqtt_server("room-data/" + roomId + "/chillCode/PeopleInRoom", number_of_people)

        # Réponse en cas de succès
        return jsonify({'status': 'success', 'numberOfPeople': number_of_people}), 200

    except Exception as e:
        # Gestion des erreurs et réponse avec le code d'erreur
        error_message = f'Error: {str(e)}'
        return jsonify({'message': error_message, 'status': 'error'}), 400

@app.route('/api/relaxWork', methods=['POST'])
def getNumberOfPeopleInARoom():
    # 40 NMS, 10 CONF
    try:
        personCount = 0
        conf_threshold = 0.1
        nms_threshold = 0.4

        # Modèle YOLO pré-entrainé
        nameWeight = "/opt/peopleDetectionAPI/Detect-people-in-a-room/yolov4.weights"
        nameCfg = "/opt/peopleDetectionAPI/Detect-people-in-a-room/yolov4.cfg"
        net = cv2.dnn.readNetFromDarknet(nameWeight, nameCfg)
        layer_names = net.getUnconnectedOutLayersNames()

        # Charger l'image
        base64_string = request.json['image']
        roomId = request.json['roomid']
        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        height, width = image.shape[:2]

        # Normaliser l'image
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Obtenir les prédictions
        outs = net.forward(layer_names)

        # Post-traitement des prédictions
        boxes = []
        confidences = []
        class_ids = []

        # Boites et leurs noms
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Supprimer les boîtes non maximales
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Dessiner les boîtes sur l'image / Compter les personnes
        for j in range(len(indices)):
            index = int(indices[j])
            if confidences[index] > 0:
                personCount += 1
                # box = boxes[index]
                # x, y, w, h = box
                # label = f"Personne: {confidences[index]:.2f}"
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        mqtt.send_to_mqtt_server("room-data/" + roomId + "/relax-work/PeopleInRoom", personCount)
        return jsonify({'status': 'success', 'personCount': personCount}), 200

    except Exception as e:
        # Gestion des erreurs et réponse avec le code d'erreur
        error_message = f'Error: {str(e)}'
        return jsonify({'message': error_message, 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(debug=True)

