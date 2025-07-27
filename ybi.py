# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# Step 2: Create an expanded and balanced dummy product dataset
# This dictionary contains more sample data to ensure the script runs without errors
# and provides a more meaningful machine learning example.
data = {
    'Product Title': [
        'Wireless Bluetooth Earbuds', 'Men\'s Cotton T-Shirt', '1.5 Ton Split AC',
        'Classic Leather Wallet', 'Smartphone 6.5 inch', 'Washing Machine Top Load',
        'Digital Wrist Watch', 'Bluetooth Portable Speaker', 'Women\'s Handbag',
        'LED Television 43 inch', 'Gaming Laptop 15.6 inch', 'Women\'s Denim Jeans',
        'Microwave Oven 20L', 'Men\'s Leather Belt', 'DSLR Camera with Kit Lens',
        'Smartwatch with HR Monitor', 'Refrigerator Double Door', 'Summer Floral Dress',
        'Polarized Sunglasses', 'Air Purifier for Home'
    ],
    'Description': [
        'Compact wireless earbuds with noise cancellation', 'Comfortable half sleeve cotton t-shirt for men',
        'Energy efficient split AC with copper condenser', 'Genuine leather wallet with multiple card slots',
        'Android smartphone with powerful processor', 'Fully automatic washing machine 6.5 kg',
        'Digital watch with alarm and waterproof design', 'Portable speaker with Bluetooth 5.0 and deep bass',
        'Stylish leather handbag for women with zip closure', 'Smart LED TV with HD display and HDMI support',
        'High performance gaming laptop with dedicated graphics', 'Comfortable stretchable denim jeans for women',
        'Convection microwave oven for baking and grilling', 'Formal leather belt with a classic buckle',
        'Entry-level DSLR camera perfect for beginners', 'Smartwatch with heart rate and sleep tracking',
        'Frost-free refrigerator with 250L capacity', 'Lightweight floral dress for casual summer wear',
        'UV protection sunglasses with a modern frame', 'HEPA filter air purifier for clean indoor air'
    ],
    'Attributes': [
        'boAt, Black, Small', 'Levi\'s, Blue, M', 'LG, White, 1.5 Ton',
        'Wildhorn, Brown', 'Redmi, Black, 128GB', 'Samsung, White, 6.5kg',
        'Fastrack, Black', 'JBL, Black', 'Caprese, Red',
        'Sony, Black, 43-inch', 'ASUS, Black, 1TB SSD', 'Zara, Blue, 32',
        'IFB, Silver, 20L', 'Tommy Hilfiger, Black', 'Canon, Black',
        'Amazfit, Grey', 'Whirlpool, Grey, 250L', 'H&M, Pink, M',
        'Ray-Ban, Black', 'Philips, White'
    ],
    'Category': [
        'Electronics', 'Clothing', 'Appliances', 'Accessories', 'Electronics',
        'Appliances', 'Accessories', 'Electronics', 'Accessories', 'Electronics',
        'Electronics', 'Clothing', 'Appliances', 'Accessories', 'Electronics',
        'Accessories', 'Appliances', 'Clothing', 'Accessories', 'Appliances'
    ]
}


# Create a pandas DataFrame from the dictionary
df = pd.DataFrame(data)

# Step 3: Combine text fields for a comprehensive input feature
# Concatenating the title, description, and attributes gives the model more
# context for each product.
df['combined_text'] = df['Product Title'] + " " + df['Description'] + " " + df['Attributes']

# Step 4: Encode categorical labels into a numerical format
# Machine learning models require numerical input, so we convert the text-based
# category names (like 'Electronics') into numbers.
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])
num_classes = len(label_encoder.classes_)

# Convert the numerical labels into a one-hot encoded format. This is necessary
# for multi-class classification with a softmax activation function.
y = to_categorical(df['label'], num_classes=num_classes)

# Step 5: Vectorize text data using TF-IDF
# TF-IDF (Term Frequency-Inverse Document Frequency) converts the raw text into
# a matrix of numerical values, representing the importance of each word.
# We limit the features to the top 1000 most frequent words.
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['combined_text']).toarray()

# Step 6: Split the data into training and testing sets
# We'll use 80% of the data to train the model and the remaining 20% to test its performance.
# Stratify ensures the distribution of categories is the same in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['label'])

# Step 7: Define the Artificial Neural Network (ANN) model
model = Sequential()
# Input layer: 128 neurons, using ReLU activation. The input dimension
# matches the number of features from the TF-IDF vectorizer.
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
# Dropout layer to prevent overfitting by randomly setting 30% of neuron outputs to zero.
model.add(Dropout(0.3))
# Hidden layer: 64 neurons with ReLU activation.
model.add(Dense(64, activation='relu'))
# Another dropout layer for regularization.
model.add(Dropout(0.2))
# Output layer: The number of neurons equals the number of categories.
# Softmax activation outputs a probability distribution across the classes.
model.add(Dense(num_classes, activation='softmax'))

# Compile the model, specifying the optimizer, loss function, and metrics.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model's architecture
print("Model Summary:")
model.summary()

# Step 8: Train the model on the training data
# The model is trained for 20 epochs with a batch size of 4.
# 20% of the training data is used for validation during training.
print("\nStarting Model Training...")
history = model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.2, verbose=1)
print("Model Training Finished.")

# Step 9: Evaluate the model's performance on the unseen test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.2f}")

# Step 10: Generate predictions on the test set
y_pred = model.predict(X_test)
# Convert the predicted probabilities into class labels (the index of the max probability).
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert the one-hot encoded true labels back to single class labels.
y_true = np.argmax(y_test, axis=1)

# Step 11: Display the classification report and confusion matrix
print("\nClassification Report:\n")
# The classification report shows precision, recall, and F1-score for each class.
print(classification_report(
    y_true,
    y_pred_classes,
    labels=np.arange(num_classes),
    target_names=label_encoder.classes_,
    zero_division=0
))

# The confusion matrix visualizes the model's performance by showing
# correct and incorrect predictions for each class.
cm = confusion_matrix(y_true, y_pred_classes, labels=np.arange(num_classes))

# Plot the confusion matrix using a heatmap for better visualization.
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
