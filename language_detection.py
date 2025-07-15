from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def load_dataset():
    """Load dataset relative to repository root."""
    # Determine path of this script and dataset location
    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / 'aaiml_proj' / 'Language Detection.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return _normalize_labels(df)


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Fix common misspellings in the language labels."""
    corrections = {
        "Portugeese": "Portuguese",
        "Sweedish": "Swedish",
    }
    df = df.copy()
    df["Language"] = df["Language"].replace(corrections)
    return df


def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'], df['Language'], test_size=0.2, random_state=42
    )
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score:.3f}")
    return model


def interactive_predict(model):
    while True:
        try:
            user_input = input("\nEnter a sentence (or 'exit'): ")
        except EOFError:
            break
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            print("Please enter some text.")
            continue
        pred = model.predict([user_input])[0]
        print("Detected Language:", pred)


if __name__ == "__main__":
    data = load_dataset()
    clf = train_model(data)
    interactive_predict(clf)
