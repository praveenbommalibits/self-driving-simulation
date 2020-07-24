"""
Step1: Loading Data
Step2: Visualization and understanding the data
Step3: Pre-processing
Step4: Splitting the data into training and testing data
Step5: Augmentation
Step 6: Pre-processing
Step 7:
Step 8: model creation
Step 9: Training model
Step 10: Saving and Plotting

Step 11: Testing

"""
print('Setting UP')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from sklearn.model_selection import train_test_split
import socketio
import eventlet
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
app = Flask(__name__)
maxSpeed = 10

#### Step 1:
path = "raw_data"
data = import_data_info(path)

### Step 2:
data = balance_data(data, display=False)

#### Step 3:
images_path, steerings = load_data(path, data)
# print(images_path[0], steering[0])

#### Step 4:
x_train, x_val, y_train, y_val = train_test_split(images_path, steerings, test_size=0.2, random_state=5)
print("Total training Images :", len(x_train))
print("Total Validation Images :", len(x_val))

#### Step 5:


#### Step 6:

####Step 7:

#### Step8:
model = create_model()
model.summary()

#### Step9:
history = model.fit(batch_generator(x_train, y_train, 100, 1), steps_per_epoch=300, epochs=10,
                    validation_data=batch_generator(x_val, y_val, 100, 0), validation_steps=200)

#### Step10:
model.save('model.h5')
print("Model Saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training ', ' Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()


def pre_process(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = pre_process(image)
    image = np.asarray([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print('{} {} {} '.format(steering, throttle, speed))
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("CONNECTED")
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
