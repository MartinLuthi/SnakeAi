# Snake Deep Q-Learning (version éducative)

Ce projet entraîne une IA à jouer à Snake avec **Deep Q-Learning (DQN)**.
Le but de cette version est de rester simple et lisible pour une personne qui débute.

## 1) Idée générale (très simple)

L’agent fait une boucle infinie :

1. Observe l’état du jeu (`state`)
2. Choisit une action (`tout droit`, `droite`, `gauche`)
3. Reçoit une récompense (`reward`)
4. Met à jour son réseau de neurones pour mieux jouer ensuite

Avec le temps, il apprend quelles actions donnent plus de récompense.

---

## 2) Structure des fichiers

- `game.py` : environnement Snake (règles, collisions, nourriture, récompenses, affichage)
- `agent.py` : agent DQN (state, mémoire replay, choix d’action)
- `model.py` : réseau de neurones + étape d’entraînement Q-learning
- `train.py` : boucle d’entraînement principale

---

## 3) Entrée du réseau (features)

La taille d’entrée est **611** :

- **11 features logiques**
  - danger tout droit
  - danger à droite
  - danger à gauche
  - direction actuelle (4 booléens)
  - position relative de la nourriture (4 booléens)
- **600 features de grille** (`30 x 20`)
  - grille complète aplatie (flatten)
  - encodage normalisé :
    - `0` = vide
    - `1/3` = corps
    - `2/3` = tête
    - `1` = nourriture

---

## 4) Récompenses utilisées

Dans `game.py` :

- `+10` quand le snake mange
- `-10` quand il meurt
- `-0.01` à chaque pas (évite de tourner en rond)
- `+0.2` s’il se rapproche de la nourriture
- `-0.2` s’il s’en éloigne

Ces signaux aident l’agent à apprendre plus vite qu’avec seulement “mort/manger”.

---

## 5) Exploration (epsilon-greedy)

L’agent explore au début, puis exploite de plus en plus :

- `epsilon = max(10, 120 - n_games)`
- Donc il garde toujours un minimum d’exploration (`10`)

---

## 6) Installation et lancement

## Prérequis

- Python 3.10+ (testé avec Python 3.14)

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install torch numpy
```

`tkinter` est souvent inclus avec Python sur Windows.

## Lancer l’entraînement

```bash
py .\train.py
```

ou

```bash
venv\Scripts\python.exe train.py
```

---

## 7) Sauvegarde du modèle

- Si tu fais `Ctrl + C`, le modèle est sauvegardé automatiquement dans `model.pth`.

---

## 8) Conseils pour apprendre en modifiant le code

1. Change une seule chose à la fois (récompense, epsilon, taille du réseau, etc.)
2. Observe les scores sur plusieurs centaines de games
3. Garde une trace de ce que tu testes

Bon apprentissage 🚀
