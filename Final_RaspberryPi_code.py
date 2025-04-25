import serial
import time
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
from collections import deque
import RPi.GPIO as GPIO
from time import sleep
from adafruit_servokit import ServoKit

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
COMMAND_PIN = 21
GPIO.setup(COMMAND_PIN, GPIO.OUT)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

interpreter = tf.lite.Interpreter(model_path='/home/picap/Desktop/m1.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scaler = joblib.load(r'/home/picap/Desktop/s1.pkl')

vote_queue = deque(maxlen=100)
label_encoder = LabelEncoder()
unique_labels = ['Rest', 'Fist', 'Paper', 'Okay']
label_encoder.fit(unique_labels)
last_voted_prediction = None

buffer = []
fs = 1070  # Sampling frequency
window_length = int(0.30 * fs)  # 300 ms window
increment = int(0.030 * fs)  # 30 ms increment
threshold = 40.90
ThumbFinger = 0
IndexFinger = 1
MiddleFinger = 2
RingFinger = 3
PinkyFinger = 4
Thumb_Extension = 5
kit = ServoKit(channels=8)
dl = 4
for i in range(5):
    kit.servo[i].set_pulse_width_range(500, 2500)
kit.servo[5].set_pulse_width_range(500, 2400)

def RestGesture():
    kit.servo[ThumbFinger].angle = 30
    kit.servo[IndexFinger].angle = 30
    kit.servo[MiddleFinger].angle = 30
    kit.servo[RingFinger].angle = 30
    kit.servo[PinkyFinger].angle = 30
    kit.servo[Thumb_Extension]=0
    time.sleep(1)

def FistGesture():
    kit.servo[ThumbFinger].angle = 180
    kit.servo[IndexFinger].angle = 180
    kit.servo[MiddleFinger].angle = 180
    kit.servo[RingFinger].angle = 180
    kit.servo[PinkyFinger].angle = 180
    kit.servo[Thumb_Extension]= 90
    time.sleep(dl)
    RestGesture()

def PaperGesture():
    kit.servo[ThumbFinger].angle = 0
    kit.servo[IndexFinger].angle = 0
    kit.servo[MiddleFinger].angle = 0
    kit.servo[RingFinger].angle = 0
    kit.servo[PinkyFinger].angle = 0
    kit.servo[ Thumb_Extension]= 90
    time.sleep(dl)
    RestGesture()

def OkayGesture():
    kit.servo[ThumbFinger].angle = 180
    kit.servo[IndexFinger].angle = 180
    kit.servo[MiddleFinger].angle = 0
    kit.servo[RingFinger].angle = 0
    kit.servo[PinkyFinger].angle = 0
    kit.servo[ThumbExtension].angle = 0
    time.sleep(dl)
    RestGesture()

def calculate_mav(segment):
    mav_sensor1 = np.mean(np.abs(segment[:, 0]))
    mav_sensor2 = np.mean(np.abs(segment[:, 1]))
    mav_sensor3 = np.mean(np.abs(segment[:, 2]))
    return (mav_sensor1 + mav_sensor2 + mav_sensor3) / 3

def calculate_features(segment):
    features = {}

    ssc_threshold = 0.01
    features['SSC Sensor1'] = ((np.diff(segment[:, 0][:-1]) * np.diff(segment[:, 0][1:]) < 0) & 
                              (np.abs(np.diff(segment[:, 0][:-1]) - np.diff(segment[:, 0][1:])) >= ssc_threshold)).sum()
    features['SSC Sensor2'] = ((np.diff(segment[:, 1][:-1]) * np.diff(segment[:, 1][1:]) < 0) & 
                              (np.abs(np.diff(segment[:, 1][:-1]) - np.diff(segment[:, 1][1:])) >= ssc_threshold)).sum()
    features['SSC Sensor3'] = ((np.diff(segment[:, 2][:-1]) * np.diff(segment[:, 2][1:]) < 0) & 
                              (np.abs(np.diff(segment[:, 2][:-1]) - np.diff(segment[:, 2][1:])) >= ssc_threshold)).sum()

    features['RMS Sensor1'] = np.sqrt(np.mean(segment[:, 0]**2))
    features['RMS Sensor2'] = np.sqrt(np.mean(segment[:, 1]**2))
    features['RMS Sensor3'] = np.sqrt(np.mean(segment[:, 2]**2))

    features['WL Sensor1'] = np.sum(np.abs(np.diff(segment[:, 0])))
    features['WL Sensor2'] = np.sum(np.abs(np.diff(segment[:, 1])))
    features['WL Sensor3'] = np.sum(np.abs(np.diff(segment[:, 2])))

    features['Skewness Sensor1'] = skew(segment[:, 0])
    features['Skewness Sensor2'] = skew(segment[:, 1])
    features['Skewness Sensor3'] = skew(segment[:, 2])

    features['Log Detector Sensor1'] = np.exp(np.mean(np.log(np.abs(segment[:, 0]) + 1e-10)))
    features['Log Detector Sensor2'] = np.exp(np.mean(np.log(np.abs(segment[:, 1]) + 1e-10)))
    features['Log Detector Sensor3'] = np.exp(np.mean(np.log(np.abs(segment[:, 2]) + 1e-10)))

    features['TM4 Sensor1'] = np.mean((segment[:, 0] - np.mean(segment[:, 0]))**4)
    features['TM4 Sensor2'] = np.mean((segment[:, 1] - np.mean(segment[:, 1]))**4)
    features['TM4 Sensor3'] = np.mean((segment[:, 2] - np.mean(segment[:, 2]))**4)

    def hjorth_params(signal):
        first_deriv = np.diff(signal)
        second_deriv = np.diff(first_deriv)
        var_zero = np.var(signal)
        var_d1 = np.var(first_deriv)
        var_d2 = np.var(second_deriv)
        activity = var_zero
        mobility = np.sqrt(var_d1 / var_zero)
        complexity = np.sqrt(var_d2 / var_d1) / mobility
        return mobility, complexity

    features['Hjorth Mobility Sensor1'], features['Hjorth Complexity Sensor1'] = hjorth_params(segment[:, 0])
    features['Hjorth Mobility Sensor2'], features['Hjorth Complexity Sensor2'] = hjorth_params(segment[:, 1])
    features['Hjorth Mobility Sensor3'], features['Hjorth Complexity Sensor3'] = hjorth_params(segment[:, 2])

    def frequency_features(signal, fs):
        f, Pxx = welch(signal, fs, nperseg=min(256, len(signal)))
        total_power = np.sum(Pxx)
        mean_freq = np.sum(f * Pxx) / total_power if total_power != 0 else 0
        return mean_freq

    features['MNF Sensor1'] = frequency_features(segment[:, 0], fs)
    features['MNF Sensor2'] = frequency_features(segment[:, 1], fs)
    features['MNF Sensor3'] = frequency_features(segment[:, 2], fs)

    return features

print("System initialized")
while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if line == "STOP":
            continue
        if line:
            sample = list(map(int, line.split(',')))
            buffer.append(sample)

        if len(buffer) >= window_length:
            segment_data = np.array(buffer[:window_length])
            buffer = buffer[increment:]  

            mav_average = calculate_mav(segment_data)

            if mav_average < threshold:
                pred_label = 'Rest'
            else:
                features = calculate_features(segment_data)
                X_test = pd.DataFrame([features]).values
                X_test_scaled = scaler.transform(X_test)

                interpreter.set_tensor(input_details[0]['index'], X_test_scaled.astype(np.float32))
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                y_pred_class = np.argmax(output_data, axis=1)[0]

                pred_label = label_encoder.inverse_transform([y_pred_class])[0]

            print(f"Predicted Gesture: {pred_label}")

            if pred_label == 'Rest':
                RestGesture()
            elif pred_label == 'Fist':
                FistGesture()
            elif pred_label == 'Paper':
                PaperGesture()
            elif pred_label == 'Okay':
                OkayGesture()

            GPIO.output(COMMAND_PIN, GPIO.LOW)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

ser.close()
GPIO.cleanup()