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
        base64_string = request.json['base64_string']
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
        mqtt.send_to_mqtt_server("room-data/chillCode/PeopleInRoom", number_of_people)

        # Réponse en cas de succès
        return jsonify({'status': 'success', 'numberOfPeople': number_of_people}), 200

    except Exception as e:
        # Gestion des erreurs et réponse avec le code d'erreur
        error_message = f'Error: {str(e)}'
        return jsonify({'message': error_message, 'status': 'error'}), 400


if __name__ == '__main__':
    app.run(debug=True)
