#%%====== SECTION 1: Project Setup ======

import numpy as np
import seaborn as sns
import tensorflow.keras as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():

    import_data()
    objective_1()

#%%====== SECTION 2: Main Execution ======
def import_data():

    global vapor_fraction
    vapor_fraction_full = np.loadtxt('./vapor_fraction_Boiling-81_110W.txt')
    vapor_fraction = vapor_fraction_full[:6000]

def data_prep(input_sequence_length, output_sequence_length, stride):

    number_of_pairs = np.shape(vapor_fraction)[0]-(input_sequence_length+output_sequence_length)+1

    input_sequence_pairs = []
    output_sequence_pairs_true = []

    for i in range(0, number_of_pairs, stride):
        input_sequence_pairs.append(vapor_fraction[i:i+input_sequence_length,1])
        output_sequence_pairs_true.append(vapor_fraction[i+input_sequence_length:i+input_sequence_length+output_sequence_length,1])

    input_sequence_pairs = np.expand_dims(input_sequence_pairs, axis=-1)
    output_sequence_pairs_true = np.expand_dims(output_sequence_pairs_true, axis=-1)

    input_train_validation, input_test, output_train_validation, output_test = train_test_split(input_sequence_pairs, output_sequence_pairs_true, test_size=0.1, random_state=42, shuffle=False)

    return input_train_validation, input_test, output_train_validation, output_test

def load_lstm(input_sequence_length, output_sequence_length):

    lstm_model = tf.models.load_model(f"lstmmodel_{input_sequence_length}_{output_sequence_length}.keras")

    return lstm_model
#%% ====== SECTION 3: Section Objectives and Plotting ======
def objective_1():

    input_sequence_length = 50
    output_sequence_length = 50
    stride = 1

    input_train_validation, input_test, output_train_validation, output_test = data_prep(input_sequence_length, output_sequence_length, stride)

    lstm_model = load_lstm(input_sequence_length, output_sequence_length)

    lstm_prediction = lstm_model.predict(input_test)

    time_step = vapor_fraction[0:100,0]

    plt.figure(figsize=(8,6))

    plt.plot(time_step[0:50,].flatten(), input_test[25].flatten(),
                color = 'Red',
                label = 'True Input Values')
    plt.scatter(time_step[50:100,].flatten(), lstm_prediction[25].flatten(),
                color = 'green',
                label = 'LSTM-Predictions',
                alpha = 0.5)
    plt.scatter(time_step[50:100,].flatten(), output_test[25].flatten(),
                color = 'blue',
                label = 'True Output Values')

    plt.xlabel('Time Step (0.33 ms)')
    plt.ylabel('Vapor Fraction')
    plt.title('Vapor Fraction Over an Arbitrary Timeframe')
    plt.legend()
    plt.show()

#%% ====== SECTION 4: Main Function Call ======
main()