import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_json('dataset.json')


def predict_temperature(date, meteo):
    df = pd.DataFrame(data)

    # Conversion des timestamps en datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Conversion des timestamps en secondes depuis l'époque Unix
    df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())

    # Sélection des caractéristiques (timestamp et peopleCount)
    X = df[['timestamp', 'peopleCount', 'outTemperature']]

    # Cible (température)
    y = df['temperature']

    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Création et entraînement du modèle de régression linéaire multiple
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction pour une nouvelle date et nombre de personnes
    new_date = pd.to_datetime(date)
    new_timestamp = new_date.timestamp()

    predictions = []
    # Création du DataFrame pour la prédiction
    for i in range(0, 12):
        df_new = pd.DataFrame({
            'timestamp': [new_timestamp],
            'peopleCount': i,
            'outTemperature': meteo
        })
        # Utiliser le modèle pour prédire la température
        predicted_temperature = model.predict(df_new)
        predictions.append(predicted_temperature)
    # return the average of all predictions
    average = sum(predictions) / len(predictions)
    return average[0]
