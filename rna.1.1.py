import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# CHARGEMENT DE DONNEES
donnees = pd.read_csv("nat2022.csv")

# Vectorisation des prenoms
vectoriseur = CountVectorizer(analyzer='char', ngram_range=(1, 2))
X = vectoriseur.fit_transform(donnees['preusuel'].astype(str)).toarray()

# classes : 1 pour fille, 0 pour garçon
y = np.array([1 if genre == 'fille' else 0 for genre in donnees['genre']])

# sep apprentissage/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# INITIALISATION : poids,biais
np.random.seed(0)
poids = np.zeros(X.shape[1])
biais = 0

# FONCTION D'ACTIVATION : binaire
def fonction_act(x):
    return 1 if x >= 0 else 0

def perceptron(x_vect):
    return fonction_act(np.dot(poids, x_vect) + biais)

## FONCTION PREDICTION
def predire_prenom(prenom):
    vect = vectoriseur.transform([prenom.lower()]).toarray()[0]
    resultat = perceptron(vect)
    return "fille" if resultat == 1 else "garçon"

# AVANT entrainement
print("AVANT ENTRAINEMENT")
print("'Olivia' est prenom de :", predire_prenom("Olivia"))
print("'Olivier' est prenom de :", predire_prenom("Olivier"))

# Entrainement
learning_rate = 0.05
nombre_epochs = 3

for epoch in range(nombre_epochs):
    erreurs = 0
    for i in range(len(X_train)):
        x_i = X_train[i]
        y_predit = perceptron(x_i)
        erreur = y_train[i] - y_predit
        poids += learning_rate * erreur * x_i
        biais += learning_rate * erreur
        erreurs += abs(erreur)
    #precision
    predictions = [perceptron(x) for x in X_test]
    precision = sum(p == t for p, t in zip(predictions, y_test)) / len(y_test)
    print(f"PENDANT ENTRAINEMENT {epoch+1}")
    print(f"Époque {epoch+1} - Erreurs : {erreurs} - Précision : {precision:.2f}")

# APRES ENTRAINEMENT
print("APRES ENTRAINEMENT")
print("'Olivia' est prenom de :", predire_prenom("Olivia"))
print("'Olivier' est prenom de :", predire_prenom("Olivier"))
