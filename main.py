import cv2
import numpy as np
from flask import Flask, jsonify, request
import base64
from yolo_segmentation import YOLOSegmentation
# Importation de la classe YOLOSegmentation depuis yolo_segmentation.py

app = Flask(__name__)


# Définition d'une route pour l'API
@app.route('/api', methods=['POST'])
def get_data():
    try:
        nmbOfPoeple = 0
        base64_string = request.json['base64_string']
        imgdata = base64.b64decode(base64_string)
        img_array = np.frombuffer(imgdata, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)


        # Initialiser le détecteur de segmentation YOLO avec le modèle "yolov8m-seg.pt"
        ys = YOLOSegmentation("yolov8m-seg.pt")

        # Effectuer la détection d'objets et obtenir les boîtes englobantes, les classes, les segments de segmentation et les scores
        bboxes, classes, segmentations, scores = ys.detect(img)

        # Parcourir les résultats de la détection pour chaque objet détecté
        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            # Déballer les coordonnées de la boîte englobante
            (x, y, x2, y2) = bbox

            # Vérifier si la classe détectée correspond à la classe d'indice 0 (remplacer par le bon ID de classe si nécessaire)
            if class_id == 0:
               nmbOfPoeple = nmbOfPoeple + 1

        return jsonify(nmbOfPoeple)

    except Exception as e:
        # En cas d'erreur lors du décodage ou du traitement
        error_message = f'Erreur : {str(e)}'
        error_data = {
            'message': error_message,
            'status': 'error'
        }
        return jsonify(error_data), 400  # Réponse avec un code d'erreur 400 Bad Request


if __name__ == '__main__':
    app.run(debug=True)
