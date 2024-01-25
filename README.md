
# ChillCode API

## Introduction
ChillCode API is a Python-based Application Programming Interface (API) designed for detecting the number of people in a base64 encoded image and predicting indoor temperature based on outdoor weather conditions. This API utilizes YOLO v4 and YOLO v8 for person detection.

## Features
1. **Person Detection:** Detects the number of people in a base64 encoded image.
2. **Indoor Temperature Prediction:** Provides temperature prediction inside based on the date and external weather.

## Endpoints

### 1. Person Detection  using YOLO v8
- **URL:** `/api/chillCode`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "roomid": "Room ID",
    "image": "base64 encoded image"
  }
  ```
- **Response:**
  ```json
  {
    "numberOfPeople": 11,
    "status": "success"
  }
  ```

### 2. Temperature Prediction using Sickit-learn
- **URL:** `/api/getTemperature`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "date": "date for prediction, e.g., '2024-01-30T15:00:00'",
    "meteo": "external weather at the predicted date, e.g., '20'"
  }
  ```
- **Response:**
  ```json
  {
    "temperature": 17.71237135060262,
    "status": "success"
  }
  ```

### 3. Person Detection using YOLO v4
- **URL:** `/api/relaxWork`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "roomid": "Room ID",
    "image": "base64 encoded image"
  }
  ```
- **Response:**
  ```json
  {
    "personCount": 4,
    "status": "success"
  }
  ```

## Usage
To use this API, send an HTTP POST request to the corresponding endpoint with the required body.

## Technologies Used
- YOLO v4 and YOLO v8 for person detection
- Sickit-learn for temperature prediction
- Python for the API backend

---
ChillCode API.
