import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "banana_detector.h5")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Make sure you have trained the model.")

model = tf.keras.models.load_model(model_path)

test_dir = os.path.join(current_dir, "test")
img_size = (100, 100)
batch_size = 32

assert os.path.exists(os.path.join(test_dir, "banana")), "Missing 'banana' folder in test dataset."
assert os.path.exists(os.path.join(test_dir, "others")), "Missing 'others' folder in test dataset."

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False  # No shuffling to maintain order
)

loss, accuracy = model.evaluate(test_generator)

print("\nModel Evaluation on Test Dataset:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

num_banana_test = len(os.listdir(os.path.join(test_dir, "banana")))
num_others_test = len(os.listdir(os.path.join(test_dir, "others")))
print(f"Number of test banana images: {num_banana_test}")
print(f"Number of test 'others' images: {num_others_test}")
print(f"Total test images: {num_banana_test + num_others_test}")
