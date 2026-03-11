import os
import numpy as np
import tensorflow as tf
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset_path = "dataset"

img_size = 224

images = []
labels = []

classes = os.listdir(dataset_path)

for class_name in classes:

    class_path = os.path.join(dataset_path,class_name)

    # FIXED IMAGE LOADING
    for root, dirs, files in os.walk(class_path):

        for file in files:

            img_path = os.path.join(root,file)

            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image,(img_size,img_size))

            images.append(image)
            labels.append(class_name)

images = np.array(images)/255.0
labels = np.array(labels)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# CNN Feature Extractor
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

feature_model = tf.keras.Model(inputs=base_model.input,outputs=x)

# Extract Features
features = feature_model.predict(images)

# Train SVM
X_train,X_test,y_train,y_test = train_test_split(features,y,test_size=0.2)

svm = SVC(kernel="rbf",probability=True)

svm.fit(X_train,y_train)

accuracy = svm.score(X_test,y_test)

print("Model Accuracy:",accuracy)

# Save Models
os.makedirs("models",exist_ok=True)

feature_model.save("models/cnn_feature_model.h5")

joblib.dump(svm,"models/svm_model.pkl")

joblib.dump(encoder,"models/label_encoder.pkl")

print("Models saved successfully")