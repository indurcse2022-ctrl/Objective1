import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

feature_model = tf.keras.Model(inputs=base_model.input, outputs=x)

feature_model.save("models/cnn_feature_model.h5")

print("CNN feature model saved successfully")