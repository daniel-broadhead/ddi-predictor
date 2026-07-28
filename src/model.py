from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb


def train_random_forest(X_train, y_train, random_state=42):
    """Random Forest baseline with balanced class weighting."""
    rf = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train, X_val, y_val, num_class, random_state=42):
    """Multiclass XGBoost with sample weighting for class imbalance
    and early stopping to avoid unnecessary training time."""
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_class,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=random_state,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=20
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """Print a full per-class report plus macro/micro F1, and
    return the metrics as a dict for later comparison/logging."""
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')

    print(report)
    print(f'Macro F1: {macro_f1:.4f}')
    print(f'Micro F1: {micro_f1:.4f}')

    return {'macro_f1': macro_f1, 'micro_f1': micro_f1, 'y_pred': y_pred, 'report': report}