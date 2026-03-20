# 🚀 Qdrant AgentPublic Indexer

Indexez les datasets [AgentPublic](https://huggingface.co/AgentPublic) de HuggingFace dans une base vectorielle [Qdrant](https://qdrant.tech/) avec **recherche hybride** (dense + sparse BM25).

## ✨ Fonctionnalités

- **Détection automatique** du champ d'embedding et de sa dimension
- **Détection automatique** du champ texte principal
- **Collections hybrides** — dense (cosine) + sparse (BM25/IDF)
- **Préservation complète du payload** — tous les champs du dataset sont stockés dans Qdrant, avec métadonnées de type
- **Indexation reprise** — système de checkpoints pour reprendre après une interruption
- **Batch adaptatif** — la taille du batch se réduit automatiquement en cas de timeouts répétés
- **Filtrage optionnel** — indexez uniquement un sous-ensemble du dataset selon la valeur d'un champ
- **Validation post-import** — rapport de taux de préservation par champ

## 📋 Prérequis

- Python 3.9+
- Une instance Qdrant accessible (locale ou cloud)
- Un compte HuggingFace (optionnel, pour les datasets privés)

### Lancer Qdrant en local avec Docker

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 🛠️ Installation

```bash
git clone https://github.com/Ktulu-Analog/index-agentpublic.git
cd index-agentpublic
pip install -r requirements.txt
```

## ⚙️ Configuration

Copiez le fichier d'exemple et renseignez vos valeurs :

```bash
cp .env.example .env
```

| Variable | Défaut | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | URL de votre instance Qdrant |
| `HF_TOKEN` | *(optionnel)* | Token HuggingFace pour les datasets privés |

## 🚀 Utilisation

```bash
python index-agentpublic.py
```

Le script est entièrement interactif et vous guide en 4 étapes :

1. **Sélection du dataset** — liste tous les datasets AgentPublic disponibles ; choisissez par numéro ou par nom
2. **Analyse automatique** — détecte le champ d'embedding, la dimension et le champ texte à partir d'un échantillon
3. **Filtrage (optionnel)** — filtrez par valeur de champ avant l'indexation
4. **Indexation** — indexe avec checkpoints, puis propose une validation de la préservation

### Exemple de session

```
🚀 INDEXATION PRODUCTION - AgentPublic → Qdrant
======================================================================
🔌 Connexion à Qdrant: http://localhost:6333
✅ Connecté (tentative 1/3)

📋 ÉTAPE 1: SÉLECTION DU DATASET
   1. albert-light         2. chunks-fr-v3  ...

Dataset (numéro ou nom): 1
✅ Dataset: AgentPublic/albert-light

📊 Détection automatique:
   ✅ Embedding: embeddings_albert (768D)
   ✅ Texte: chunk_text

Appliquer un filtre ? (oui/non): non

Lancer ? (oui/non): oui
🚀 Indexation de 1 234 567 documents...
```

## 🗂️ Structure du projet

```
qdrant-agentpublic-indexer/
├── index_agent_public.py   # Script principal d'indexation
├── requirements.txt        # Dépendances Python
├── .env.example            # Modèle de variables d'environnement
├── .gitignore              # Règles Git ignore
└── README.md               # Ce fichier
```

Les checkpoints sont sauvegardés dans `.checkpoints/<nom_collection>.json` et exclus de git.

## 🔍 Fonctionnement

### Recherche hybride

Chaque point Qdrant contient deux vecteurs :

- **`dense`** — l'embedding float extrait du dataset (ex. `embeddings_albert`)
- **`sparse`** — un vecteur BM25 calculé à la volée sur le champ texte via `Qdrant/bm25`

Cela permet la recherche hybride (similarité dense + correspondance par mots-clés) sans configuration supplémentaire.

### Sérialisation du payload

Tous les champs du dataset sont stockés dans le payload avec une fidélité de type complète :

| Type dataset | Payload Qdrant |
|---|---|
| `str`, `int`, `float`, `bool` | valeur native |
| `list` / `dict` | chaîne JSON |
| `None` | `null` |

Une clé de métadonnée `_type_<champ>` est ajoutée à côté de chaque champ pour faciliter la désérialisation.

### Checkpoints

Un checkpoint est sauvegardé toutes les `CHECKPOINT_INTERVAL` secondes (défaut : 5 000 s) et lors d'un `KeyboardInterrupt`. Au prochain lancement, le script propose de reprendre depuis le dernier index sauvegardé.

## ⚡ Réglage des performances

Modifiez les constantes dans la classe `Config` :

```python
BATCH_SIZE = 100           # Taille initiale des batches d'upload
CHECKPOINT_INTERVAL = 5000 # Secondes entre les checkpoints automatiques
MIN_BATCH_SIZE = 10        # Plancher pour la réduction dynamique du batch
```

Le script divise automatiquement la taille du batch par 2 après 3 timeouts consécutifs, jusqu'à `MIN_BATCH_SIZE`.

---

## 📄 Licence

Ce projet est distribué sous licence **AGPL-3.0**.  
Voir [https://www.gnu.org/licenses/agpl-3.0.html](https://www.gnu.org/licenses/agpl-3.0.html).

---

## 👤 Auteur

Pierre COUGET — 2026
