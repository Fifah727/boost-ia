import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

# CHARGEMENT DE DONNEES
donnees = pd.read_csv("nat2022.csv") 

# Vectorisation des prenom (lettres + bigrammes)
vectoriseur = CountVectorizer(analyzer='char', ngram_range=(1, 2))
X = vectoriseur.fit_transform(donnees['preusuel'].astype(str)).toarray()

# Création des etiquettes : 0 pour garçon, 1 pour filles
y = np.array([1 if genre == 'fille' else 0 for genre in donnees['genre']])

# Sep en donnees d'entrainement et de tests
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Def du modèle TensorFlow
modele = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sortie binaire (0 ou 1)
])

# Compilation du modèle
modele.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Evaluation AVANT entrainement
_, precision = modele.evaluate(X_test, y_test, verbose=0)
print(f"[AVANT ENTRAÎNEMENT] Précision : {precision:.2f}")

# Entraînement du modèle
modele.fit(X_train, y_train, epochs=3, batch_size=32)

#Evaluation APRES entrainement
_, precision = modele.evaluate(X_test, y_test, verbose=0)
print(f"[APRÈS ENTRAÎNEMENT] Précision : {precision:.2f}")

# Fonction de prediction
def predire_prenom(prenom):
    vecteur = vectoriseur.transform([prenom.lower()]).toarray()
    prediction = modele.predict(vecteur, verbose=0)[0][0]
    return "Fille" if prediction >= 0.5 else "Garçon"

print("'Olivia' prédit comme :", predire_prenom("Olivia"))
print("'Olivier' prédit comme :", predire_prenom("Olivier"))
