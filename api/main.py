from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO #Just like what we do with variables, data can be kept as bytes in an in-memory buffer when we use the io module’s Byte IO operations.
from PIL import Image # used to read images in python
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model('./models/1')
CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  #converting image to numpy ass
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
#how to convert file into numpy array
    image = read_file_as_image(await file.read())

#[256,256,3]

    img_batch = np.expand_dims(image, 0)  #Insert a new axis that will appear at the axis position in the expanded array shape.

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    
    confidence = np.max(predictions[0])

    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)