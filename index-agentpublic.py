#!/usr/bin/env python3
# ============================================================================
# SCRIPT D'INDEXATION PRODUCTION - AgentPublic → Qdrant
#
# Usage: python index-agentpublic.py
# ============================================================================
# Auteur  : Pierre COUGET
# Licence : GNU Affero General Public License v3.0 (AGPL-3.0)
#           https://www.gnu.org/licenses/agpl-3.0.html
# Année   : 2026
# ----------------------------------------------------------------------------
# Vous pouvez le redistribuer et/ou le modifier selon les termes de la
# licence AGPL-3.0 publiée par la Free Software Foundation.
# ============================================================================

import os
import time
import json
import ast
import random
from typing import Dict, Any, List, Optional
from pathlib import Path
from threading import Lock

# charge .env pour récupérer les paramètres
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from datasets import load_dataset
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from huggingface_hub import HfApi

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration globale."""

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_TIMEOUT = 120

    # HuggingFace
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN")

    # Performance
    BATCH_SIZE = 100
    CHECKPOINT_INTERVAL = 5000
    MIN_BATCH_SIZE = 10  # Taille minimale en cas de timeout

    # Validation
    POST_VALIDATION_SAMPLE = 100


# ============================================================================
# UTILITAIRES
# ============================================================================

def print_columns(items: List[str], num_columns: int = 2, col_width: int = 30, numbered: bool = True):
    """Affiche une liste en colonnes."""
    num_items = len(items)
    rows = (num_items + num_columns - 1) // num_columns

    for row in range(rows):
        line_parts = []
        for col in range(num_columns):
            idx = row + col * rows
            if idx < num_items:
                if numbered:
                    item_str = f"  {idx + 1:2d}. {items[idx]:{col_width}s}"
                else:
                    item_str = f"  {items[idx]:{col_width}s}"
                line_parts.append(item_str)
            else:
                line_parts.append(" " * (col_width + 6))
        print("".join(line_parts))


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Gestion des checkpoints."""

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.lock = Lock()

    def get_path(self, collection: str) -> Path:
        return self.checkpoint_dir / f"{collection}.json"

    def save(self, collection: str, data: Dict[str, Any]) -> None:
        with self.lock:
            path = self.get_path(collection)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    def load(self, collection: str) -> Optional[Dict[str, Any]]:
        path = self.get_path(collection)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Erreur lecture checkpoint: {e}")
        return None


# ============================================================================
# ANALYSEUR DE DATASET
# ============================================================================

class DatasetAnalyzer:
    """Analyse automatique des datasets."""

    @staticmethod
    def list_agentpublic_datasets() -> List[str]:
        """Liste tous les datasets AgentPublic."""
        try:
            api = HfApi()
            datasets = list(api.list_datasets(author="AgentPublic"))
            return [ds.id for ds in datasets]
        except Exception as e:
            print(f"⚠️  Erreur: {e}")
            return []

    @staticmethod
    def detect_embedding_field(sample: Dict[str, Any]) -> tuple:
        """Détecte le champ d'embedding et sa dimension."""
        for key, value in sample.items():
            if hasattr(value, 'as_py'):
                value = value.as_py()

            # Liste directe
            if isinstance(value, (list, tuple)) and len(value) > 100:
                try:
                    if isinstance(value[0], (float, int)):
                        dimension = len(value)
                        model_name = key.replace("embeddings_", "").replace("embedding_", "")
                        return key, dimension, model_name
                except:
                    pass

            # String stringifiée
            elif isinstance(value, str) and value.startswith('['):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list) and len(parsed) > 100:
                        if isinstance(parsed[0], (int, float)):
                            dimension = len(parsed)
                            model_name = key.replace("embeddings_", "").replace("embedding_", "")
                            return key, dimension, model_name
                except:
                    pass

        return None, None, None

    @staticmethod
    def detect_text_field(sample: Dict[str, Any], exclude_keys: List[str] = None) -> Optional[str]:
        """Détecte le champ texte principal."""
        if exclude_keys is None:
            exclude_keys = []

        text_candidates = []

        for key, value in sample.items():
            if key in exclude_keys:
                continue

            if hasattr(value, 'as_py'):
                value = value.as_py()

            if isinstance(value, str) and not value.startswith('['):
                if len(value) > 50:
                    priority = 0
                    if "chunk_text" in key.lower():
                        priority = 3
                    elif "text" in key.lower():
                        priority = 2
                    elif "content" in key.lower():
                        priority = 1

                    text_candidates.append((key, len(value), priority))

        if not text_candidates:
            return None

        text_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return text_candidates[0][0]

    @staticmethod
    def analyze_field_values(dataset, field_name: str, max_unique: int = 100) -> Dict[str, int]:
        """Analyse les valeurs uniques d'un champ."""
        print(f"\n🔍 Analyse du champ '{field_name}'...")

        value_counts = {}
        sample_size = min(len(dataset), 10000)

        with tqdm(total=sample_size, desc="Échantillonnage", unit="docs") as pbar:
            for i in range(sample_size):
                item = dataset[i]
                value = item.get(field_name)

                if value is not None:
                    if hasattr(value, 'as_py'):
                        value = value.as_py()
                    value_str = str(value)
                    value_counts[value_str] = value_counts.get(value_str, 0) + 1

                pbar.update(1)

                if len(value_counts) > max_unique:
                    print(f"⚠️  Plus de {max_unique} valeurs uniques")
                    return {}

        return dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True))


# ============================================================================
# INDEXEUR AVEC PRÉSERVATION 100%
# ============================================================================

class ProductionIndexer:
    """Indexeur production avec préservation du contenu."""

    def __init__(self):
        self.client = None
        self.checkpoint_mgr = CheckpointManager()
        self.config = None
        self.current_batch_size = Config.BATCH_SIZE  # Taille dynamique
        self.consecutive_timeouts = 0  # Compteur de timeouts
        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "none_values": {},
            "timeouts": 0,
            "retries": 0
        }
        self.stats_lock = Lock()

    def connect(self) -> bool:
        """Connexion à Qdrant avec retry."""
        print(f"\n🔌 Connexion à Qdrant: {Config.QDRANT_URL}")

        for attempt in range(1, 4):
            try:
                self.client = QdrantClient(url=Config.QDRANT_URL, timeout=Config.QDRANT_TIMEOUT)
                collections = self.client.get_collections()
                print(f"✅ Connecté (tentative {attempt}/3)")
                print(f"   Collections existantes: {len(collections.collections)}")
                return True
            except Exception as e:
                print(f"⚠️  Tentative {attempt}/3 échouée: {e}")
                if attempt < 3:
                    time.sleep(attempt * 2)

        print("\n❌ Connexion impossible")
        print("💡 Vérifiez: docker ps | grep qdrant")
        return False

    def create_collection(self, collection_name: str, dimension: int) -> bool:
        """Crée une collection hybride."""
        print(f"\n📦 Création collection '{collection_name}'...")

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                timeout=Config.QDRANT_TIMEOUT
            )
            print(f"✅ Collection créée")
            print(f"   Dense: {dimension}D, Distance: Cosine")
            print(f"   Sparse: BM25 avec IDF modifier")
            return True
        except Exception as e:
            print(f"❌ Erreur création: {e}")
            return False

    def serialize_value(self, value) -> tuple:
        """Sérialise une valeur pour Qdrant."""
        if hasattr(value, 'as_py'):
            value = value.as_py()

        if value is None:
            return None, "null"

        if isinstance(value, bool):
            return value, "bool"
        if isinstance(value, int):
            return value, "int"
        if isinstance(value, float):
            return value, "float"
        if isinstance(value, str):
            return value, "str"

        # ✅ Listes/tuples → JSON
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "[]", "json_list"
            try:
                return json.dumps(value, ensure_ascii=False), "json_list"
            except:
                return str(value), "str_list"

        # ✅ Dicts → JSON
        if isinstance(value, dict):
            if len(value) == 0:
                return "{}", "json_dict"
            try:
                return json.dumps(value, ensure_ascii=False), "json_dict"
            except:
                return str(value), "str_dict"

        # Autres → String
        try:
            return str(value), "str_other"
        except:
            return None, "null"

    def extract_complete_payload(self, item: dict, idx: int, exclude_fields: list) -> dict:
        """Extrait un payload complet avec TOUS les champs."""
        payload = {"_index": idx}

        for key, value in item.items():
            if key in exclude_fields:
                continue

            # Tracker les None
            if value is None or (hasattr(value, 'as_py') and value.as_py() is None):
                if key not in self.stats["none_values"]:
                    self.stats["none_values"][key] = 0
                self.stats["none_values"][key] += 1

            # Sérialiser
            serialized, type_info = self.serialize_value(value)

            # ✅ TOUJOURS inclure, même None
            payload[key] = serialized
            payload[f"_type_{key}"] = type_info

        return payload

    def upload_batch_with_retry(self, collection_name: str, points: List[models.PointStruct], max_retries: int = 3) -> bool:
        """Upload un batch avec retry en cas de timeout."""

        for attempt in range(1, max_retries + 1):
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True  # Attendre la confirmation
                )

                # ✅ Succès - Réinitialiser compteur
                self.consecutive_timeouts = 0
                return True

            except Exception as e:
                error_msg = str(e).lower()

                if "timeout" in error_msg or "timed out" in error_msg:
                    with self.stats_lock:
                        self.stats["timeouts"] += 1
                        self.stats["retries"] += 1

                    self.consecutive_timeouts += 1

                    # Adapter la taille du batch si trop de timeouts
                    if self.consecutive_timeouts >= 3 and self.current_batch_size > Config.MIN_BATCH_SIZE:
                        self.current_batch_size = max(Config.MIN_BATCH_SIZE, self.current_batch_size // 2)
                        print(f"\n⚠️  Timeouts répétés - Réduction batch: {self.current_batch_size}")

                    if attempt < max_retries:
                        # Diviser le batch en cas de timeout
                        if len(points) > 10:
                            print(f"\n⚠️  Timeout ({len(points)} points) - Division du batch...")
                            mid = len(points) // 2

                            # Diviser en 2 et réessayer
                            if self.upload_batch_with_retry(collection_name, points[:mid], max_retries=2):
                                if self.upload_batch_with_retry(collection_name, points[mid:], max_retries=2):
                                    return True
                            return False
                        else:
                            print(f"\n⚠️  Timeout - Retry {attempt}/{max_retries}...")
                            time.sleep(attempt * 3)  # Backoff progressif
                    else:
                        print(f"\n❌ Échec upload après {max_retries} tentatives ({len(points)} points)")
                        return False
                else:
                    # Autre erreur
                    print(f"\n❌ Erreur upload: {e}")
                    with self.stats_lock:
                        self.stats["retries"] += 1

                    if attempt < max_retries:
                        time.sleep(2)
                    else:
                        return False

        return False
        """Crée une collection hybride."""
        print(f"\n📦 Création collection '{collection_name}'...")

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                timeout=Config.QDRANT_TIMEOUT
            )
            print("✅ Collection créée")
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False

    def prepare_point(self, item: dict, idx: int, embedding_field: str, text_field: str, dimension: int) -> Optional[models.PointStruct]:
        """Prépare un point Qdrant."""
        try:
            # Texte
            text = str(item.get(text_field, ""))
            if len(text) < 10:
                return None

            # Embedding
            emb_value = item.get(embedding_field)
            if hasattr(emb_value, 'as_py'):
                emb_value = emb_value.as_py()

            if isinstance(emb_value, str):
                embedding = ast.literal_eval(emb_value)
            else:
                embedding = list(emb_value)

            if len(embedding) != dimension:
                return None

            # ✅ Payload complet
            payload = self.extract_complete_payload(item, idx, [embedding_field])

            # Point
            return models.PointStruct(
                id=idx,
                vector={
                    "dense": embedding,
                    "sparse": models.Document(text=text, model="Qdrant/bm25")
                },
                payload=payload
            )

        except Exception as e:
            return None

    def index_dataset(self, dataset, start_idx: int, collection_name: str, embedding_field: str, text_field: str, dimension: int):
        """Indexe le dataset avec checkpoints."""

        total = len(dataset)
        print(f"\n🚀 Indexation de {total - start_idx:,} documents...")

        # Désactiver indexing HNSW
        try:
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0)
            )
            print("⚙️  Indexation HNSW désactivée")
        except:
            pass

        start_time = time.time()
        last_checkpoint = start_time
        points_buffer = []

        try:
            with tqdm(total=total - start_idx, desc="Indexation", unit="docs") as pbar:
                for idx in range(start_idx, total):
                    item = dataset[idx]
                    point = self.prepare_point(item, idx, embedding_field, text_field, dimension)

                    with self.stats_lock:
                        self.stats["processed"] += 1

                    if point:
                        points_buffer.append(point)
                        with self.stats_lock:
                            self.stats["success"] += 1
                    else:
                        with self.stats_lock:
                            self.stats["failed"] += 1

                    # Upload batch avec retry
                    if len(points_buffer) >= self.current_batch_size:
                        success = self.upload_batch_with_retry(collection_name, points_buffer)
                        if not success:
                            # Sauvegarder et arrêter
                            print("\n⚠️  Erreurs répétées - sauvegarde checkpoint...")
                            self.save_checkpoint(collection_name, idx, total)
                            print("💡 Relancez le script pour reprendre")
                            return time.time() - start_time
                        points_buffer.clear()

                    pbar.update(1)

                    # Checkpoint
                    if time.time() - last_checkpoint > Config.CHECKPOINT_INTERVAL:
                        self.save_checkpoint(collection_name, idx, total)
                        last_checkpoint = time.time()

                # Upload reste
                if points_buffer:
                    self.upload_batch_with_retry(collection_name, points_buffer)

        except KeyboardInterrupt:
            print("\n⚠️  Interruption - sauvegarde checkpoint...")
            self.save_checkpoint(collection_name, self.stats["processed"], total)
            raise

        # Réactiver HNSW
        try:
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
            )
            print("\n✅ Indexation HNSW réactivée")
        except:
            pass

        # Checkpoint final
        self.save_checkpoint(collection_name, total, total, final=True)

        elapsed = time.time() - start_time
        return elapsed

    def save_checkpoint(self, collection: str, current: int, total: int, final: bool = False):
        """Sauvegarde checkpoint."""
        data = {
            "collection": collection,
            "last_index": current,
            "total": total,
            "stats": self.stats.copy(),
            "timestamp": time.time(),
            "is_final": final
        }
        self.checkpoint_mgr.save(collection, data)

    def validate_preservation(self, dataset, collection_name: str, embedding_field: str, sample_size: int = 100):
        """Valide la préservation des données."""
        print("\n" + "=" * 70)
        print("🔍 VALIDATION POST-IMPORT")
        print("=" * 70)

        dataset_size = len(dataset)
        check_size = min(sample_size, dataset_size)
        indices = random.sample(range(dataset_size), check_size)

        print(f"Vérification de {check_size} points...")

        field_comparison = {}

        for idx in tqdm(indices, desc="Validation"):
            try:
                result = self.client.retrieve(
                    collection_name=collection_name,
                    ids=[idx],
                    with_payload=True
                )

                if result:
                    qdrant_point = result[0]
                    dataset_item = dataset[idx]

                    for key in dataset_item.keys():
                        if key == embedding_field:
                            continue

                        if key not in field_comparison:
                            field_comparison[key] = {"present": 0, "absent": 0}

                        if key in qdrant_point.payload:
                            field_comparison[key]["present"] += 1
                        else:
                            field_comparison[key]["absent"] += 1
            except:
                pass

        # Rapport
        print("\n📊 TAUX DE PRÉSERVATION PAR CHAMP:")
        print("-" * 70)

        total_preserved = 0
        total_fields = 0

        for field, counts in sorted(field_comparison.items()):
            total = counts["present"] + counts["absent"]
            rate = (counts["present"] / total * 100) if total > 0 else 0

            status = "✅" if rate == 100 else "⚠️" if rate > 50 else "❌"
            print(f"{status} {field:30s}: {rate:5.1f}% ({counts['present']}/{total})")

            total_preserved += counts["present"]
            total_fields += total

        print("-" * 70)
        overall_rate = (total_preserved / total_fields * 100) if total_fields > 0 else 0
        print(f"{'📊 TOTAL':32s}: {overall_rate:5.1f}% ({total_preserved}/{total_fields})")

    def print_report(self, elapsed: float):
        """Rapport final."""
        print("\n" + "=" * 70)
        print("📊 RÉSULTATS FINAUX")
        print("=" * 70)
        print(f"✅ Traités:  {self.stats['processed']:,}")
        print(f"✅ Indexés:  {self.stats['success']:,}")
        print(f"❌ Échecs:   {self.stats['failed']:,}")

        if self.stats.get('timeouts', 0) > 0:
            print(f"⚠️  Timeouts: {self.stats['timeouts']:,}")
            print(f"🔄 Retries:  {self.stats['retries']:,}")

        print(f"⏱️  Temps:    {elapsed/60:.1f} min")

        if self.stats['processed'] > 0:
            speed = self.stats['processed'] / elapsed
            print(f"⚡ Vitesse:  {speed:.0f} docs/s")

        if self.current_batch_size < Config.BATCH_SIZE:
            print(f"\n📦 Taille batch finale: {self.current_batch_size} (réduite de {Config.BATCH_SIZE})")

        if self.stats["none_values"]:
            print(f"\n📊 Champs avec valeurs None:")
            for field, count in sorted(self.stats["none_values"].items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / self.stats['processed'] * 100)
                print(f"   {field:30s}: {count:6,} ({percentage:5.1f}%)")

        print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Point d'entrée principal."""

    print("=" * 70)
    print("🚀 INDEXATION PRODUCTION - AgentPublic → Qdrant")
    print("=" * 70)

    analyzer = DatasetAnalyzer()
    indexer = ProductionIndexer()

    # Connexion
    if not indexer.connect():
        return

    # ==================================================
    # ÉTAPE 1: Sélection dataset
    # ==================================================

    print("\n📋 ÉTAPE 1: SÉLECTION DU DATASET")
    print("=" * 70)

    available_datasets = analyzer.list_agentpublic_datasets()

    if available_datasets:
        print(f"\n✅ {len(available_datasets)} dataset(s):\n")
        short_names = [ds.replace("AgentPublic/", "") for ds in available_datasets]
        print_columns(short_names, num_columns=2, col_width=35)

    dataset_input = input("\nDataset (numéro ou nom): ").strip()

    # Résoudre
    if dataset_input.isdigit():
        idx = int(dataset_input) - 1
        if 0 <= idx < len(available_datasets):
            dataset_name = available_datasets[idx]
        else:
            print("❌ Numéro invalide")
            return
    elif dataset_input:
        if not dataset_input.startswith("AgentPublic/"):
            dataset_input = f"AgentPublic/{dataset_input}"
        dataset_name = dataset_input
    else:
        print("❌ Aucun dataset")
        return

    print(f"✅ Dataset: {dataset_name}")

    # Nom collection
    default_collection = dataset_name.replace("AgentPublic/", "").replace("/", "_")
    collection_name = input(f"Collection (défaut: {default_collection}): ").strip() or default_collection

    # ==================================================
    # ÉTAPE 2: Analyse
    # ==================================================

    print("\n📋 ÉTAPE 2: ANALYSE DU DATASET")
    print("=" * 70)

    print("\n📥 Chargement échantillon...")
    dataset_sample = load_dataset(dataset_name, split="train[:10]")
    sample = dataset_sample[0]

    # Détecter champs
    embedding_field, dimension, model_name = analyzer.detect_embedding_field(sample)
    text_field = analyzer.detect_text_field(sample, [embedding_field] if embedding_field else [])

    print(f"\n📊 Détection automatique:")
    if embedding_field:
        print(f"   ✅ Embedding: {embedding_field} ({dimension}D)")
    else:
        print(f"   ❌ Embedding non détecté")
        return

    if text_field:
        print(f"   ✅ Texte: {text_field}")
    else:
        print(f"   ❌ Texte non détecté")
        return

    # ==================================================
    # ÉTAPE 3: Filtrage (optionnel)
    # ==================================================

    print("\n📋 ÉTAPE 3: FILTRAGE (OPTIONNEL)")
    print("=" * 70)

    all_fields = list(sample.keys())
    filter_field = None
    filter_values = None

    want_filter = input("\nAppliquer un filtre ? (oui/non): ").strip().lower()

    if want_filter in ["oui", "o", "yes", "y"]:
        print("\nChamps disponibles:")
        print_columns(all_fields[:30], num_columns=3, col_width=22)

        filter_choice = input("\nChamp à filtrer: ").strip()

        # Résoudre le choix (numéro ou nom)
        if filter_choice.isdigit():
            idx = int(filter_choice) - 1
            if 0 <= idx < len(all_fields):
                filter_choice = all_fields[idx]
                print(f"   → Champ sélectionné: {filter_choice}")
            else:
                print(f"❌ Numéro invalide (1-{len(all_fields)})")
                filter_choice = None
        elif filter_choice not in all_fields:
            print(f"❌ Champ '{filter_choice}' introuvable")
            filter_choice = None

        if filter_choice:
            # Charger plus de données pour analyse
            print("\n📥 Chargement dataset complet...")
            dataset_full = load_dataset(dataset_name, split="train")

            values = analyzer.analyze_field_values(dataset_full, filter_choice)

            if values:
                print(f"\n📊 Valeurs du champ '{filter_choice}':")
                print("-" * 70)

                total_docs = len(dataset_full)
                for i, (value, count) in enumerate(list(values.items())[:20], 1):
                    percentage = (count / total_docs) * 100
                    print(f"  {i:2d}. {value:40s}: {count:>10,} ({percentage:5.1f}%)")

                if len(values) > 20:
                    print(f"  ... et {len(values) - 20} autres valeurs")

                print("-" * 70)

                print("\n💡 Options:")
                print("  - Tapez les valeurs séparées par des virgules")
                print("  - Tapez 'tout' pour ne pas filtrer")
                print("  - Tapez les numéros séparés par des virgules (ex: 1,2,5)")

                values_input = input("\nVotre sélection: ").strip()

                if values_input and values_input.lower() != 'tout':
                    # Vérifier si ce sont des numéros
                    if all(part.strip().isdigit() for part in values_input.split(",")):
                        # Numéros → Résoudre en valeurs
                        selected_values = []
                        values_list = list(values.keys())

                        for num_str in values_input.split(","):
                            idx = int(num_str.strip()) - 1
                            if 0 <= idx < len(values_list):
                                selected_values.append(values_list[idx])
                            else:
                                print(f"⚠️  Numéro {num_str} ignoré (invalide)")

                        if selected_values:
                            filter_field = filter_choice
                            filter_values = selected_values
                            print(f"✅ Filtre: {filter_field} = {filter_values}")
                    else:
                        # Valeurs directes
                        filter_field = filter_choice
                        filter_values = [v.strip() for v in values_input.split(",")]
                        print(f"✅ Filtre: {filter_field} = {filter_values}")
            else:
                print(f"⚠️  Trop de valeurs uniques dans '{filter_choice}' pour filtrage")
                dataset_full = load_dataset(dataset_name, split="train")
        else:
            dataset_full = load_dataset(dataset_name, split="train")
    else:
        dataset_full = load_dataset(dataset_name, split="train")

    # Filtrer
    if filter_field and filter_values:
        print(f"\n🔍 Filtrage: {filter_field} = {filter_values}")
        initial_count = len(dataset_full)

        dataset = dataset_full.filter(
            lambda x: str(x.get(filter_field)) in filter_values,
            desc="Filtrage"
        )

        print(f"✅ {len(dataset):,} / {initial_count:,} documents")
    else:
        dataset = dataset_full

    # ==================================================
    # ÉTAPE 4: Création collection
    # ==================================================

    # Vérifier si existe
    collections = indexer.client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)

    if collection_exists:
        print(f"\n⚠️  Collection '{collection_name}' existe")
        action = input("Action (r=recréer, c=continuer, a=annuler): ").strip().lower()

        if action == "r":
            indexer.client.delete_collection(collection_name)
            print("✅ Supprimée")
            collection_exists = False
        elif action != "c":
            print("❌ Annulé")
            return

    if not collection_exists:
        if not indexer.create_collection(collection_name, dimension):
            return

    # Checkpoint
    checkpoint = indexer.checkpoint_mgr.load(collection_name)
    start_idx = 0

    if checkpoint and not checkpoint.get("is_final"):
        last_idx = checkpoint.get("last_index", 0)
        print(f"\n📋 Checkpoint: {last_idx:,} traités")

        resume = input("Reprendre ? (oui/non): ").strip().lower()
        if resume in ["oui", "o", "yes", "y", ""]:
            start_idx = last_idx

    # ==================================================
    # RÉSUMÉ
    # ==================================================

    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    print(f"Dataset:     {dataset_name}")
    print(f"Collection:  {collection_name}")
    print(f"Documents:   {len(dataset):,}")
    print(f"À indexer:   {len(dataset) - start_idx:,}")
    print(f"Embedding:   {embedding_field} ({dimension}D)")
    print(f"Texte:       {text_field}")
    if filter_field:
        print(f"Filtre:      {filter_field} = {filter_values}")
    print("=" * 70)

    confirm = input("\nLancer ? (oui/non): ").strip().lower()
    if confirm not in ["oui", "o", "yes", "y", ""]:
        print("❌ Annulé")
        return

    # ==================================================
    # INDEXATION
    # ==================================================

    elapsed = indexer.index_dataset(
        dataset, start_idx, collection_name,
        embedding_field, text_field, dimension
    )

    # ==================================================
    # VALIDATION
    # ==================================================

    validate = input("\nValider la préservation ? (oui/non): ").strip().lower()
    if validate in ["oui", "o", "yes", "y", ""]:
        indexer.validate_preservation(
            dataset, collection_name, embedding_field,
            sample_size=min(100, len(dataset))
        )

    # ==================================================
    # RAPPORT
    # ==================================================

    indexer.print_report(elapsed)

    print("\n✅ Indexation terminée !")
    print(f"\n💡 Utilisation:")
    print(f'   collection="{collection_name}"')
    print(f'   method="hybrid"')


if __name__ == "__main__":
    main()
