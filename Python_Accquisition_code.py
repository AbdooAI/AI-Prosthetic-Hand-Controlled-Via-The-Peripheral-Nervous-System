import serial
import time
import csv
from time import sleep

# Set the serial port and baud rate
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Initialize the serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

class colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'

# Function to get user input for person ID and gender
def get_user_info():
    person_id = input("Enter the Person ID: ")
    gender = input("Enter the Gender (Male/Female): ")
    return person_id, gender

# Initialize the error counter
error_count = 0

# Define a function to record data for a specified duration with a stopwatch
def record_period(duration, label, round_number, gesture_name, writer, person_id, gender):
    global error_count
    print(f"{colors.GREEN}Start in 2 seconds{colors.RESET}")
    sleep(2)  # Give time to prepare before recording starts
    print(f"{colors.RED}Recording {label} for {gesture_name}, Round {round_number}.{colors.RESET}")

    # Send start command to Arduino
    ser.write(b'START\n')

    start_time = time.time()  # Record the start time

    while True:
        line = ser.readline().decode('utf-8').strip()
        if line == "STOP":
            break
        if line:
            sensorValues = line.split(',')
            if len(sensorValues) >= 3:
                try:
                    sensorValue1 = int(sensorValues[0])
                    sensorValue2 = int(sensorValues[1])
                    sensorValue3 = int(sensorValues[2])
                    elapsed_time = time.time() - start_time  # Calculate elapsed time
                    elapsed_time_str = f"{elapsed_time:.6f}"  # Format elapsed time to 6 decimal places for microseconds
                    writer.writerow([elapsed_time_str, sensorValue1, sensorValue2, sensorValue3, label, round_number, gesture_name, person_id, gender])
                except ValueError:
                    print("Error parsing sensor values: Skipping this line.")
                    error_count += 1  # Increment the error counter
            else:
                print("Incomplete data received, skipping...")
        else:
            print("Empty line received, skipping...")

def main():
    global error_count
    # Open a CSV file to store the data
    person_id, gender = get_user_info()
    with open('emg_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Elapsed Time (s)', 'Sensor1', 'Sensor2', 'Sensor3', 'Label', 'Round', 'Gesture', 'Person ID', 'Gender'])
        gestures = ['Fist','Thumb Finger','Index Finger','Scissors','Paper','Three','Four','Okay','Finger Gun'] 
        GESTURE_DURATION = 5  # duration for each gesture recording in seconds
        SHORT_REST = 10  # short rest between rounds in seconds

        for gesture_name in gestures:
            input(f"Press Enter to start sequence for {gesture_name} or Ctrl+C to exit.")
            for round_number in range(1, 5):
                for _ in range(5):
                    record_period(GESTURE_DURATION, 'Rest', round_number, 'Rest', writer, person_id, gender)
                    
                    record_period(GESTURE_DURATION, 'Non-Rest', round_number, gesture_name, writer, person_id, gender)

                if round_number < 4:
                    print(f"Short rest of {SHORT_REST} seconds.")
                    sleep(SHORT_REST)

            if gesture_name != gestures[-1]:
                print("Long rest of 3 minutes until next gesture.")
                sleep(120)
                print("1 min Remaining")
                sleep(60)

        print("Data collection complete for all gestures. Processing data...")
        print(".....")
        print("Rename CSV File for current patient so the data doesn't get overwritten with the next patient's data")
        print(f"Total parsing errors encountered: {error_count}")

if _name_ == "_main_":
    main()    
