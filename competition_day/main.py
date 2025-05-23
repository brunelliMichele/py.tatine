# main del progetto, da qui partono le chiamate ai vari modelli
import subprocess
import os

def choose_and_run_model():
    while True:
        print("Scegli il modello da eseguire: ")
        print("0 - Esci")
        print("1 - CLIP")
        print("2 - DINO")
        print("3 - VGG16")
        print("4 - ResNet50")
        choice = input("Inserisci il numero corrispondente: ").strip()

        if choice == "0":
            print("👋 Uscita.")
            break
        if choice == "1":
            script_path = os.path.join("CLIP", "main_clip_retrieval.py")
        elif choice == "2":
            script_path = os.path.join("DinoV2", "main_dino_retrieval.py")
        elif choice == "3":
            script_path = os.path.join("VGG16", "main_vgg16_retrieval.py")
        elif choice == "4":
            script_path = os.path.join("ResNet", "main_resnet50_retrieval.py")
        else:
            print("❌ Scelta non valida.")
            continue
        
        if os.path.exists(script_path):
            subprocess.run(["python", script_path], check=True)
        else:
            print(f"❌ Script non trovato: {script_path}")

if __name__ == "__main__":
    choose_and_run_model()