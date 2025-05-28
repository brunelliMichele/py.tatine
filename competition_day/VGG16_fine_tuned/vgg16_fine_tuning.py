import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Percorso dei dati
train_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')

print(f"📁 Verifica cartella di training: {train_dir}")
# Parametri
img_size = (224, 224)
batch_size = 32
num_epochs = 10
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"La cartella '{train_dir}' non esiste.")

subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
if not subdirs:
    raise RuntimeError(f"La cartella '{train_dir}' non contiene sottocartelle di classi.")

num_classes = len(subdirs)

print("🧪 Preparo generatori di immagini con data augmentation...")
# Data augmentation e preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
) 

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print("📥 Caricamento modello VGG16 base (senza top)...")
# Carica VGG16 senza top
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
print("🔧 Modello VGG16 modificato con classifier custom.")

print("❄️ Congelo i layer convoluzionali di VGG16...")
# Congela i layer convoluzionali
for layer in base_model.layers:
    layer.trainable = False

print("⚙️ Compilo il modello...")
# Compila il modello
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print(f"💾 Il modello migliore sarà salvato in: vgg16_finetuned_from_script.keras")
# Salvataggio del modello
checkpoint = ModelCheckpoint('vgg16_finetuned_from_script.keras', save_best_only=True, monitor='val_accuracy', mode='max')

print("🚀 Inizio fase di allenamento...")
# Allenamento
model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    callbacks=[checkpoint]
)
print("✅ Allenamento completato!")