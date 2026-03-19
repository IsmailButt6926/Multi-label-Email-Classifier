# This is the main file: The controller.
# All methods will directly or indirectly be called here.
#
# It implements both design choices:
#   Design Choice 1: Chained Multi-Output Classification
#   Design Choice 2: Hierarchical Modelling
#
# The results of both are printed and compared at the end.

from preprocess import *
from embeddings import *
from modelling.modelling import chained_model_predict, hierarchical_model_predict
from modelling.data_model import *
import random
import numpy as np

seed = Config.SEED
random.seed(seed)
np.random.seed(seed)


def load_data():
    """Load the input data from CSV files."""
    df = get_input_data()
    return df


def preprocess_data(df):
    """Apply all preprocessing steps to the data."""
    # De-duplicate input data
    df = de_duplication(df)
    # Remove noise in input data
    df = noise_remover(df)
    # Translate data to english
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings_data(df):
    """Convert text data to TF-IDF embeddings."""
    X = get_tfidf_embd(df)
    return X, df


# ======================================================================
# Code execution starts here
# ======================================================================
if __name__ == '__main__':

    # ==================================================================
    # PHASE 1: PREPROCESSING
    # ==================================================================
    print("=" * 70)
    print("  PHASE 1: Loading and Preprocessing Data")
    print("=" * 70)

    df = load_data()
    print(f"  Loaded {len(df)} records from {len(Config.DATA_FILES)} file(s)")

    df = preprocess_data(df)
    print(f"  After preprocessing: {len(df)} records")

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Show data overview
    print(f"\n  Data overview:")
    for col in ['y1', 'y2', 'y3', 'y4']:
        n_valid = df[col].notna().sum()
        n_classes = df[col].dropna().nunique()
        print(f"    {col}: {n_valid} valid / {len(df)} total, {n_classes} unique classes")

    # ==================================================================
    # PHASE 2: CREATE CHAINED TARGET COLUMNS (for Design Choice 1)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: Creating Chained Target Columns")
    print("=" * 70)

    df = create_chained_columns(df)

    for chain_name, cols in Config.CHAINED_TARGETS.items():
        col_name = 'y_' + chain_name
        valid = df[col_name].notna().sum()
        n_classes = df[col_name].dropna().nunique()
        print(f"  {col_name} ({' + '.join(cols)}): {valid} valid, {n_classes} unique combined labels")

    # ==================================================================
    # PHASE 3: DATA TRANSFORMATION (EMBEDDINGS)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: Computing TF-IDF Embeddings")
    print("=" * 70)

    X, group_df = get_embeddings_data(df)
    print(f"  Embedding matrix shape: {X.shape}")

    # ==================================================================
    # PHASE 4: DESIGN CHOICE 1 — CHAINED MULTI-OUTPUT CLASSIFICATION
    # ==================================================================
    dc1_results = chained_model_predict(X, df)

    # ==================================================================
    # PHASE 5: DESIGN CHOICE 2 — HIERARCHICAL MODELLING
    # ==================================================================
    dc2_results, dc2_model_count = hierarchical_model_predict(X, df)

    # ==================================================================
    # PHASE 6: COMPARISON OF BOTH DESIGN CHOICES
    # ==================================================================
    print("\n" + "*" * 70)
    print("*" + " " * 68 + "*")
    print("*  FINAL COMPARISON: Design Choice 1 vs Design Choice 2" + " " * 13 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    print("\n  DESIGN CHOICE 1 (Chained Multi-Output):")
    print(f"  {'─' * 50}")
    print(f"  Total models: {len(dc1_results)} (one RF per chain level)")
    for chain_name, info in dc1_results.items():
        print(f"  {chain_name}: {info['label']:<30} Accuracy = {info['accuracy']:.4f}")

    print(f"\n  DESIGN CHOICE 2 (Hierarchical Modelling):")
    print(f"  {'─' * 50}")
    print(f"  Total models: {dc2_model_count} (branching per class at each hierarchy level)")
    for r in dc2_results:
        if r['accuracy'] is not None:
            print(f"  Level {r['level']} | {r['parent']:<35} -> {r['target']}: Accuracy = {r['accuracy']:.4f}")
        else:
            print(f"  Level {r['level']} | {r['parent']:<35} -> {r['target']}: {r.get('note', 'N/A')}")

    print(f"\n  KEY DIFFERENCES:")
    print(f"  {'─' * 50}")
    print(f"  DC1 uses {len(dc1_results)} model(s); DC2 uses {dc2_model_count} model(s).")
    print(f"  DC1 evaluates combined labels as single targets.")
    print(f"  DC2 evaluates each type per filtered subset, giving per-class insight.")
    print(f"\n{'=' * 70}")
    print(f"  All results printed. Pipeline complete.")
    print(f"{'=' * 70}")
