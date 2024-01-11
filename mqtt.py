import random
import ssl
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

load_dotenv()  # Charge les variables d'environnement à partir de .env

mqtt_username = os.getenv('mqtt.username')
mqtt_password = os.getenv('mqtt.password')

# Utilisez maintenant `api_key` et `database_url` comme vous le souhaitez
# Paramètres MQTT
broker_address = "mqtt.freezlex.dev"  # Remplacez avec l'adresse de votre broker MQTT
broker_port = 1883  # Port standard MQTT, changez si nécessaire
username = mqtt_username
password = mqtt_password
client_id = f'publish-{random.randint(0, 1000)}'


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)


def on_publish(client, userdata, mid):
    print("Data published!")


def send_to_mqtt_server(topic, numberOfPeople):
    client = mqtt.Client(client_id)
    client.on_connect = on_connect
    client.on_publish = on_publish

    client.username_pw_set(username, password)
    client.tls_set(certfile=None, keyfile=None, cert_reqs=ssl.CERT_NONE, tls_version=ssl.PROTOCOL_TLS)
    try:
        client.connect(broker_address, broker_port)
        print("Connecting to broker...")
        client.loop_start()
        result = client.publish(topic, numberOfPeople)
        status = result[0]
        if status == 0:
            print(f"Send `{numberOfPeople}` to topic `{topic}`!")
        else:
            print(f"Failed to send message to topic {topic}")
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        print(f"Failed to connect or publish to MQTT broker due to error: {str(e)}")

