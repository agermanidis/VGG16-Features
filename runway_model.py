from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import runway

@runway.setup
def setup():
  base_model = VGG16(weights='imagenet')
  model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
  return model

@runway.command('extract_features', inputs={'image': runway.image}, outputs={'features': runway.vector(4096)})
def extract_features(model, inputs):
  x = np.array(inputs['image'].resize((224, 224)))
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  features = model.predict(x)[0]
  return features.flatten()

if __name__ == "__main__":
  runway.run()
