from PIL import Image
import numpy as np

def preprocessing_image(image: Image.Image):

    # handles all types of images greyscale , 3 channel ,4 channel ,etc
    image = image.convert("RGB")

    #  changing space of image
    image = image.resize((224,224))

    # array conversion of image
    image=np.array(image,dtype=np.float32)

    # normalizing pixel value
    image=image/255.0

    # adding batch dimension
    image=np.expand_dims(image,axis=0)

    return image