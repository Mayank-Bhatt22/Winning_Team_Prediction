# ==========================================
# IPL Match Winner Prediction - ANN Model
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("synthetic_match_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ==========================================
# 2. Select Input Features
# ==========================================

cat_cols = [
    'team1',
    'team2',
    'venue',
    'toss_winner',
    'toss_decision'
]

num_cols = [
    'team1_runs_last_5',
    'team2_runs_last_5'
]

features = cat_cols + num_cols


# ==========================================
# 3. Encode Categorical Columns
# ==========================================

le_dict = {}

for col in cat_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col].astype(str))

    le_dict[col] = le


# Encode winner
le_winner = LabelEncoder()

df['winner_encoded'] = le_winner.fit_transform(
    df['winner'].astype(str)
)


# ==========================================
# 4. Create X and y
# ==========================================

X = df[features].values

y = df['winner_encoded'].values


print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# ==========================================
# 5. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Scale Numerical Features
# ==========================================

scaler = StandardScaler()

X_train[:, -2:] = scaler.fit_transform(
    X_train[:, -2:]
)

X_test[:, -2:] = scaler.transform(
    X_test[:, -2:]
)


# ==========================================
# 7. Convert Target to One-Hot Encoding
# ==========================================

num_classes = len(le_winner.classes_)

y_train_cat = to_categorical(
    y_train,
    num_classes=num_classes
)

y_test_cat = to_categorical(
    y_test,
    num_classes=num_classes
)


# ==========================================
# 8. Build ANN Model
# ==========================================

model = Sequential()

model.add(
    Dense(
        64,
        activation='relu',
        input_shape=(X_train.shape[1],)
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        32,
        activation='relu'
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        num_classes,
        activation='softmax'
    )
)


# ==========================================
# 9. Compile Model
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ==========================================
# 10. Show Model Structure
# ==========================================

model.summary()


# ==========================================
# 11. Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=32,
    verbose=1
)


# ==========================================
# 12. Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)


# ==========================================
# 13. Classification Report
# ==========================================

y_pred_probability = model.predict(X_test)

y_pred = np.argmax(
    y_pred_probability,
    axis=1
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=le_winner.classes_,
        zero_division=0
    )
)


# ==========================================
# 14. Confusion Matrix
# ==========================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 15. Save Model
# ==========================================

model.save("ipl_winner_model.keras")


# ==========================================
# 16. Save Encoders
# ==========================================

import pickle

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(le_dict, f)

with open("winner_encoder.pkl", "wb") as f:
    pickle.dump(le_winner, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


print("\nModel and preprocessing files saved successfully!")

