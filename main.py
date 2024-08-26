import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

# Image dimensions
image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

# Classifier class
class Classifier:
    def __init__(self):
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def load(self, path):
        self.model.load_weights(path)

    def predict_single_image(self, image_path):
        # Load and preprocess the image
        img = load_img(image_path, target_size=(image_dimensions['height'], image_dimensions['width']))
        img_array = img_to_array(img) / 255.0  # Rescale image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict using the model
        confidence = self.model.predict(img_array)[0][0]
        return confidence

# Meso4 model class
class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    def init_model(self):
        x = Input(shape=(image_dimensions['height'], image_dimensions['width'], image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

# Instantiate the Meso4 model and load weights
meso = Meso4()
meso.load('./weights/Meso4_DF')

# Predict on a single image
image_path = './sample_image.jpg'  # Replace with your image path
confidence = meso.predict_single_image(image_path)

# Print the model's confidence score
print(f"Model confidence: {confidence:.4f}")
