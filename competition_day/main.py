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
        print("5 - Efficientnet")
        print("6 - ResNetFinetunnato")
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
        elif choice == "5":
            print("Scegli quale EfficientNet vuoi: ")
            print("0 - Esci")
            print("1 - EfficientNetB0")
            print("2 - EfficientNetB4")
            print("3 - merged EfficientNet")
            choice2 = input("Inserisci il numero corrispondente: ").strip()
            if choice2 == "0":
                print("👋 Uscita.")
                break
            elif choice2 == "1":
                script_path = os.path.join("image-retrieval-with-efficientnet", "EfficientNetB0.py")
            elif choice2 == "2":
                script_path = os.path.join("image-retrieval-with-efficientnet", "EfficientNetB4.py")
            elif choice2 == "3":
                script_path = os.path.join("image-retrieval-with-efficientnet", "merged_efficientNet.py")
        elif choice == "6":
            print("Scegli quale modello di ResNet far partire: ")
            print("0 - Esci")
            print("1 - L2 with Cross Entropy")
            print("2 - Tripletloss Hard Negative Mining")
            print("3 - Tripletloss")
            choice3 = input("Inserisci il numero corrispondente: ").strip()
            if choice3 == "0":
                print("👋 Uscita.")
                break
            elif choice3 == "1":
                script_path = os.path.join("ResNet-Fine-Tuning", "L2-with-CrossEntropy.py")
            elif choice3 == "2":
                script_path = os.path.join("ResNet-Fine-Tuning", "tripletloss-hard-negative-mining.py")
            elif choice3 == "3":
                script_path = os.path.join("ResNet-Fine-Tuning", "tripletloss.py")
            print("❌ Scelta non valida.")
            continue
        
        if os.path.exists(script_path):
            subprocess.run(["python", script_path], check=True)
        else:
            print(f"❌ Script non trovato: {script_path}")

if __name__ == "__main__":
    choose_and_run_model()