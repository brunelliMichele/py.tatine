# main del progetto, da qui partono le chiamate ai vari modelli
import subprocess
import os

def choose_and_run_model():
    print("Scegli il modello da eseguire: ")
    print("1 - CLIP")
    print("2 - DINO")
    choice = input("Inserisci il numero corrispondente: ").strip()

    if choice == "1":
        script_path = os.path.join("CLIP", "main_clip_retrieval.py")
    elif choice == "2":
        script_path = os.path.join("DinoV2", "main_dino_retrieval.py")
    else:
        print("❌ Scelta non valida.")
        return
    
    if os.path.exists(script_path):
        subprocess.run(["python", script_path], check=True)
    else:
        print(f"❌ Script non trovato: {script_path}")

if __name__ == "__main__":
    choose_and_run_model()