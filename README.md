
# Prédiction du genre à partir d’un prénom avec un Perceptron

Ce projet propose une approche simple de **classification binaire** basée sur un **perceptron** pour prédire le **genre** (fille ou garçon) d’un prénom, en utilisant des n-grammes de caractères pour représenter les prénoms.

## 🧠 Objectif

L’objectif est d’entraîner un perceptron (sans bibliothèque de deep learning) pour apprendre à distinguer les prénoms masculins et féminins à partir d’un ensemble de données réelles.

---

## 📁 Données

Le fichier d’entrée est :

- `nat2022.csv` : un fichier CSV contenant deux colonnes principales :
  - `preusuel` : le prénom
  - `genre` : le genre associé (`fille` ou `garçon`)

Exemple :

| preusuel | genre  |
|----------|--------|
| Emma     | fille  |
| Léo      | garçon |

---

## ⚙️ Description du fonctionnement

### 1. Chargement des données

Les données sont chargées avec `pandas`.

```python
donnees = pd.read_csv("nat2022.csv")
````

### 2. Prétraitement

* Les prénoms sont vectorisés en n-grammes de caractères (1 et 2 caractères) via `CountVectorizer`.
* Le genre est transformé en valeur binaire : `1` pour fille, `0` pour garçon.

```python
vectoriseur = CountVectorizer(analyzer='char', ngram_range=(1, 2))
X = vectoriseur.fit_transform(donnees['preusuel'].astype(str)).toarray()
y = np.array([1 if genre == 'fille' else 0 for genre in donnees['genre']])
```

### 3. Séparation des données

On sépare les données en ensemble d'entraînement et de test (80% / 20%).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Perceptron

* Fonction d’activation : seuil à zéro.
* Poids et biais initialisés à zéro.
* Apprentissage sur plusieurs époques avec une règle de mise à jour simple.

```python
def fonction_act(x):
    return 1 if x >= 0 else 0
```

---

## 🔁 Entraînement

Le perceptron est entraîné pendant plusieurs époques (`nombre_epochs = 3`) :

```python
for epoch in range(nombre_epochs):
    for i in range(len(X_train)):
        # Prédiction, calcul de l'erreur
        # Mise à jour des poids
```

---

## 🔍 Prédiction

Fonction de prédiction personnalisée :

```python
def predire_prenom(prenom):
    ...
    return "fille" ou "garçon"
```

---

## 📈 Exemple de sortie

### Avant apprentissage

```
'Olivia' est prenom de : garçon
'Olivier' est prenom de : garçon
```

### Après apprentissage

```
'Olivia' est prenom de : fille
'Olivier' est prenom de : garçon
```

---

## 📦 Dépendances

* Python ≥ 3.7
* numpy
* pandas
* scikit-learn

Installation :

```bash
pip install numpy pandas scikit-learn
```

---

## ▶️ Lancer le script

Place le fichier `nat2022.csv` dans le même dossier que le script, puis exécute :

```bash
python script.py
```

---

## 📚 Remarques

* Le modèle est très simple et peut être amélioré avec :

  * Plus de données
  * D’autres techniques de vectorisation (TF-IDF, embeddings)
  * Des modèles plus avancés (réseaux de neurones, SVM, etc.)
* L’approche reste pédagogique pour comprendre les bases de la classification et du perceptron.

---

## 👩‍💻 Auteur

**Fifaliana Sarobidy**
DA2I L3 033I23 • EMIT, 2025

---

