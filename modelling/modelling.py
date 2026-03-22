from model.randomforest import RandomForest
from modelling.data_model import Data, FilteredData
from Config import Config
from utils import remove_low_frequency_classes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =============================================================================
# SHARED HELPER
# =============================================================================

def _train_and_evaluate(data, model_name):
    """Instantiate a RandomForest, train, predict, and evaluate. Returns the model."""
    rf = RandomForest(
        model_name=model_name,
        embeddings=data.embeddings,
        y=data.y
    )
    rf.train(data)
    rf.predict(data.X_test)
    rf.print_results(data)
    return rf


# =============================================================================
# DESIGN CHOICE 1: CHAINED MULTI-OUTPUT CLASSIFICATION
# =============================================================================

def chained_model_predict(X, df):
    """Design Choice 1 — Chained Multi-Output Classification.

    For each chain level, the combined label (e.g., y2 + y3) is treated as a
    single multi-class target. One RandomForest per chain level.

    The accuracy of chain N+1 is guaranteed to be <= accuracy of chain N,
    because a correct combined label requires all constituent types to be correct.
    """
    print("\n" + "=" * 70)
    print("  DESIGN CHOICE 1: CHAINED MULTI-OUTPUT CLASSIFICATION")
    print("=" * 70)
    print("Each chain level treats the combined label as a single target.")
    print("Accuracy of chain N+1 <= Accuracy of chain N.\n")

    results = {}

    for chain_name, cols in Config.CHAINED_TARGETS.items():
        target_col = 'y_' + chain_name
        label = Config.CHAIN_SEPARATOR.join(cols)

        print(f"{'#' * 70}")
        print(f"  Chain Level: {chain_name}  |  Classifying: {label}")
        print(f"{'#' * 70}")

        # Create Data object for this chain level
        data = Data(X, df, target_col)

        print(f"  Training samples: {len(data.y_train)}, Test samples: {len(data.y_test)}")
        print(f"  Number of classes: {len(set(data.y_train))}")
        print()

        # Train and evaluate
        model = _train_and_evaluate(data, f'RF_{chain_name}')

        acc = accuracy_score(data.y_test, model.predictions)
        results[chain_name] = {
            'label': label,
            'accuracy': acc,
            'train_size': len(data.y_train),
            'test_size': len(data.y_test),
            'n_classes': len(set(data.y_train)),
        }

    # Print comparison summary
    print("\n" + "=" * 70)
    print("  DESIGN CHOICE 1 — SUMMARY")
    print("=" * 70)
    print(f"  {'Chain':<12} {'Target':<30} {'Accuracy':>10} {'Classes':>10}")
    print(f"  {'-'*12} {'-'*30} {'-'*10} {'-'*10}")
    for chain_name, info in results.items():
        print(f"  {chain_name:<12} {info['label']:<30} {info['accuracy']:>10.4f} {info['n_classes']:>10}")
    print()

    return results


# =============================================================================
# DESIGN CHOICE 2: HIERARCHICAL MODELLING
# =============================================================================

def hierarchical_model_predict(X, df):
    """Design Choice 2 — Hierarchical Modelling.

    Models are chained in a tree structure:
      Level 1: RF classifies Type 2 on the full dataset.
      Level 2: For EACH class in Type 2, a separate RF classifies Type 3
               using only the rows belonging to that Type 2 class.
      Level 3: For EACH (Type2 class, Type3 class) pair, a separate RF
               classifies Type 4 using only the matching rows.

    This allows per-class effectiveness assessment at each hierarchy level.
    """
    levels = Config.HIERARCHICAL_LEVELS  # ['y2', 'y3', 'y4']

    print("\n" + "=" * 70)
    print("  DESIGN CHOICE 2: HIERARCHICAL MODELLING")
    print("=" * 70)
    print("Models are chained in a tree: output of level N filters input for level N+1.")
    print(f"Hierarchy: {' -> '.join(levels)}")
    print(f"Total models = 1 + |classes in y2| + |classes in y2| x |classes in y3|\n")

    all_results = []
    model_count = 0

    # ---- LEVEL 1: Classify y2 on entire dataset ---------------------------
    level_col = levels[0]  # 'y2'
    print(f"{'#' * 70}")
    print(f"  LEVEL 1: Classifying {level_col} (full dataset)")
    print(f"{'#' * 70}")

    # Prepare: drop NaN in y2, remove low-freq classes
    mask_valid = df[level_col].notna()
    X_l1 = X[mask_valid.values]
    df_l1 = df[mask_valid].reset_index(drop=True)
    X_l1, df_l1 = remove_low_frequency_classes(df_l1, X_l1, level_col, Config.MIN_CLASS_COUNT)

    y_l1 = df_l1[level_col].values
    indices = np.arange(len(df_l1))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=Config.SEED, stratify=y_l1
    )

    data_l1 = FilteredData(
        X_l1[train_idx], X_l1[test_idx],
        y_l1[train_idx], y_l1[test_idx]
    )
    df_l1_train = df_l1.iloc[train_idx].reset_index(drop=True)
    df_l1_test = df_l1.iloc[test_idx].reset_index(drop=True)

    print(f"  Training: {len(data_l1.y_train)}, Test: {len(data_l1.y_test)}")
    print(f"  Classes: {sorted(set(y_l1))}")
    print()

    rf_l1 = _train_and_evaluate(data_l1, f'RF_Level1_{level_col}')
    model_count += 1

    acc_l1 = accuracy_score(data_l1.y_test, rf_l1.predictions)
    all_results.append({
        'level': 1, 'parent': 'ALL', 'target': level_col,
        'accuracy': acc_l1, 'train': len(data_l1.y_train), 'test': len(data_l1.y_test),
        'classes': sorted(set(y_l1))
    })

    # ---- LEVEL 2: For each class in y2, classify y3 -----------------------
    if len(levels) >= 2:
        level2_col = levels[1]  # 'y3'
        classes_l1 = sorted(set(y_l1))

        print(f"\n{'#' * 70}")
        print(f"  LEVEL 2: Classifying {level2_col} per {level_col} class")
        print(f"  Creating {len(classes_l1)} sub-model(s) — one per {level_col} class")
        print(f"{'#' * 70}")

        for cls in classes_l1:
            print(f"\n  --- {level_col} = '{cls}' → Classifying {level2_col} ---")

            # Filter data to rows of this class
            cls_mask = df_l1[level_col] == cls
            df_cls = df_l1[cls_mask].reset_index(drop=True)
            X_cls = X_l1[cls_mask.values]

            # Drop rows where level2 target is NaN
            valid_l2 = df_cls[level2_col].notna()
            df_cls = df_cls[valid_l2].reset_index(drop=True)
            X_cls = X_cls[valid_l2.values]

            if len(df_cls) < Config.MIN_CLASS_COUNT * 2:
                print(f"  SKIPPED: Too few samples ({len(df_cls)}) for {level_col}='{cls}'")
                all_results.append({
                    'level': 2, 'parent': f"{level_col}={cls}", 'target': level2_col,
                    'accuracy': None, 'train': 0, 'test': 0,
                    'classes': [], 'note': 'skipped — too few samples'
                })
                continue

            # Remove low-frequency classes within this subset
            X_cls, df_cls = remove_low_frequency_classes(df_cls, X_cls, level2_col, max(2, Config.MIN_CLASS_COUNT // 2))

            y_cls = df_cls[level2_col].values
            unique_classes = sorted(set(y_cls))

            if len(unique_classes) < 2:
                print(f"  SKIPPED: Only {len(unique_classes)} class(es) for {level_col}='{cls}' -> {unique_classes}")
                all_results.append({
                    'level': 2, 'parent': f"{level_col}={cls}", 'target': level2_col,
                    'accuracy': None, 'train': len(y_cls), 'test': 0,
                    'classes': unique_classes, 'note': 'skipped — single class'
                })
                continue

            # Split
            idx2 = np.arange(len(df_cls))
            tr2, te2 = train_test_split(idx2, test_size=0.2, random_state=Config.SEED, stratify=y_cls)

            data_l2 = FilteredData(X_cls[tr2], X_cls[te2], y_cls[tr2], y_cls[te2])
            print(f"  Training: {len(data_l2.y_train)}, Test: {len(data_l2.y_test)}, Classes: {unique_classes}")
            print()

            rf_l2 = _train_and_evaluate(data_l2, f'RF_Level2_{level_col}={cls}_{level2_col}')
            model_count += 1

            acc_l2 = accuracy_score(data_l2.y_test, rf_l2.predictions)
            all_results.append({
                'level': 2, 'parent': f"{level_col}={cls}", 'target': level2_col,
                'accuracy': acc_l2, 'train': len(data_l2.y_train), 'test': len(data_l2.y_test),
                'classes': unique_classes
            })

            # ---- LEVEL 3: For each class in y3 (within this y2), classify y4 ----
            if len(levels) >= 3:
                level3_col = levels[2]  # 'y4'
                classes_l2 = unique_classes

                for cls2 in classes_l2:
                    print(f"\n    --- {level_col}='{cls}', {level2_col}='{cls2}' → Classifying {level3_col} ---")

                    cls2_mask = df_cls[level2_col] == cls2
                    df_cls2 = df_cls[cls2_mask].reset_index(drop=True)
                    X_cls2 = X_cls[cls2_mask.values]

                    # Drop NaN in level3
                    valid_l3 = df_cls2[level3_col].notna()
                    df_cls2 = df_cls2[valid_l3].reset_index(drop=True)
                    X_cls2 = X_cls2[valid_l3.values]

                    if len(df_cls2) < 4:
                        print(f"    SKIPPED: Too few samples ({len(df_cls2)})")
                        all_results.append({
                            'level': 3,
                            'parent': f"{level_col}={cls}, {level2_col}={cls2}",
                            'target': level3_col,
                            'accuracy': None, 'train': 0, 'test': 0,
                            'classes': [], 'note': 'skipped — too few samples'
                        })
                        continue

                    # Remove single-instance classes
                    X_cls2, df_cls2 = remove_low_frequency_classes(df_cls2, X_cls2, level3_col, 2)
                    y_cls2 = df_cls2[level3_col].values
                    unique3 = sorted(set(y_cls2))

                    if len(unique3) < 2:
                        print(f"    SKIPPED: Only {len(unique3)} class(es) -> {unique3}")
                        all_results.append({
                            'level': 3,
                            'parent': f"{level_col}={cls}, {level2_col}={cls2}",
                            'target': level3_col,
                            'accuracy': None, 'train': len(y_cls2), 'test': 0,
                            'classes': unique3, 'note': 'skipped — single class'
                        })
                        continue

                    idx3 = np.arange(len(df_cls2))
                    tr3, te3 = train_test_split(idx3, test_size=0.2, random_state=Config.SEED, stratify=y_cls2)

                    data_l3 = FilteredData(X_cls2[tr3], X_cls2[te3], y_cls2[tr3], y_cls2[te3])
                    print(f"    Training: {len(data_l3.y_train)}, Test: {len(data_l3.y_test)}, Classes: {unique3}")
                    print()

                    rf_l3 = _train_and_evaluate(data_l3, f'RF_Level3_{level_col}={cls}_{level2_col}={cls2}_{level3_col}')
                    model_count += 1

                    acc_l3 = accuracy_score(data_l3.y_test, rf_l3.predictions)
                    all_results.append({
                        'level': 3,
                        'parent': f"{level_col}={cls}, {level2_col}={cls2}",
                        'target': level3_col,
                        'accuracy': acc_l3,
                        'train': len(data_l3.y_train), 'test': len(data_l3.y_test),
                        'classes': unique3
                    })

    # ---- RESULTS SUMMARY ---------------------------------------------------
    print("\n" + "=" * 70)
    print("  DESIGN CHOICE 2 — SUMMARY")
    print("=" * 70)
    print(f"  Total models created: {model_count}\n")
    print(f"  {'Level':<6} {'Parent Filter':<40} {'Target':<8} {'Accuracy':>10} {'Train':>7} {'Test':>6}")
    print(f"  {'-'*6} {'-'*40} {'-'*8} {'-'*10} {'-'*7} {'-'*6}")
    for r in all_results:
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else r.get('note', 'N/A')
        print(f"  {r['level']:<6} {r['parent']:<40} {r['target']:<8} {acc_str:>10} {r['train']:>7} {r['test']:>6}")
    print()

    return all_results, model_count
