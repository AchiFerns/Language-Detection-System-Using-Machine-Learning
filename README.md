# Language-Detection-System-Using-Machine-Learning
A supervised machine learning model is trained using a Kaggle dataset with text samples in 20 languages. TF-IDF vectorization is used to convert text into features, and a Multinomial Naive Bayes classifier predicts the language of user-inputted text.

## Usage

Install the Python dependencies and run the script to train the model and test it interactively:

```bash
pip install -r requirements.txt
python language_detection.py
```

For a simple web UI built with Streamlit run:

```bash
streamlit run streamlit_app.py
```

The script automatically loads the dataset from `aaiml_proj/Language Detection.csv` and fixes a few misspelled language names (e.g. `Portugeese` -> `Portuguese`). After training, it prints the accuracy score and waits for your input sentences.
