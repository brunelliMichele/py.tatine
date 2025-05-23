# -*- coding: utf-8 -*-

# lanciare questo script per inviare i risultati
import requests
import json

API_URL = "http://127.0.0.1:8000/submit"  # qui metteremo l'enpoint che il prof darà il giorno della competition
SUBMISSION_FILE = "submission.json"

def submit_file():
    with open(SUBMISSION_FILE, "r") as f:
        data = json.load(f)
    
    response = requests.post(API_URL, json = data)

    if response.status_code == 200:
        print("✅ Invio riuscito!")
        print("📊 Risposta dal server:", response.json())
    else:
        print("❌ Invio fallito:", response.status_code)
        print(response.text)

if __name__ == "__main__":
    submit_file()