from ultralytics import YOLO  # Importation du module YOLO de Ultralytics
import numpy as np  # Importation du module numpy


class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Initialisation du modèle YOLO avec le chemin du modèle passé en paramètre

    def detect(self, img):
        # Obtenir les dimensions de l'image
        height, width, channels = img.shape

        # Prédiction du modèle YOLO sur l'image d'entrée
        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]  # Récupération du résultat de la prédiction

        segmentation_contours_idx = []  # Liste pour stocker les contours de segmentation
        for seg in result.masks.xy:
            # Conversion des coordonnées normalisées des contours en pixels
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)  # Conversion en tableau numpy d'entiers
            segmentation_contours_idx.append(segment)  # Ajout du contour à la liste

        # Récupération des boîtes englobantes (bounding boxes) sous forme de tableau numpy d'entiers
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

        # Récupération des IDs de classe sous forme de tableau numpy d'entiers
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")

        # Récupération des scores de confiance sous forme de tableau numpy de nombres flottants arrondis à 2 décimales
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        # Retourne les boîtes englobantes, les IDs de classe, les contours de segmentation et les scores de confiance
        return bboxes, class_ids, segmentation_contours_idx, scores
