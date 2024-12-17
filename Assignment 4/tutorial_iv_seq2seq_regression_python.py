#%%====== SECTION 1: Project Setup ======

    # The data file “vapor_fraction_Boiling-81_110W.txt” includes the vapor fraction (second column, dimensionless) vs. time (first column, unit: ms) of the boiling image sequences. The data are sampled with a frequency of 3,000 Hz (namely, a time step of 0.33 ms). Develop a recurrent neural network (RNN) model to forecast vapor fraction of future frames based on the past frames, e.g., predicting the vapor fraction profile of t = 33.33 ms – 66 ms using the vapor fraction history of t = 0.33 – 33 ms. Options include regular RNN, bidirectional RNN, gated recurrent unit (GRU), bidirectional GRU, long short-term memory (LSTM), bidirectional LSTM.

#%%=======================
# 1a. Import Libraries
# ========================
import numpy as np
import seaborn as sns
import tensorflow.keras as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#%%=======================
# 1b. Main Function
# ========================
def main():

    import_data()
    objective_1()
    objective_2()

#%%====== SECTION 2: Main Execution ======

    #This section builds the main functions in order to perform the (2) objectives given by the homework assignment. The main sections are as follows:
        #a. Import the data from the given text file.
        #b. Prepare the data by creating sequence pairs for future analysis and splitting the data into a training/validation set and a testing set.
        #c. Build the LSTM model.
        #d. Loads a saved model based on the best performing model given by the trained LSTM model. This section is for troubleshooting only. It saves time by loading an existing model, rather than training the model every time the program is ran.
#%%=======================
# 2a. Import Data

    # This section imports the data from the text file given by the assignment. After that, it creates an array of the first 6000 values (Index# 0-5999). This is a global variable, so nothing needs to be returned from the sub-function.
# ========================
def import_data():

    global vapor_fraction
    vapor_fraction_full = np.loadtxt('./vapor_fraction_Boiling-81_110W.txt')
    vapor_fraction = vapor_fraction_full[:6000]

#%%=======================
# 2b. Create Sequence Pairs and Prepare Data for Training/Validation/Testing

    # This is vital in an LSTM model, as it allows for the prediction of the immediate next sequence given the prior sequence. Each sequence pair serves as the input and output values of the model. For example, a sequence pair with input/output length of 50 and a stride of 5 would look like the below:
    # Input: input_sequence_pair[0] = (vapor_fraction[0], vapor_fraction[1], vapor_fraction[2], ..., vapor_fraction[49])
    # Input: input_sequence_pair[1] = (vapor_fraction[5], vapor_fraction[6], vapor_fraction[7], ..., vapor_fraction[54])
    # Output: output_sequence_pair[0] = (vapor_fraction[50], vapor_fraction[51], vapor_fraction[52], ..., vapor_fraction[99])
    # Output: output_sequence_pair[1] = (vapor_fraction[55], vapor_fraction[56], vapor_fraction[57], ..., vapor_fraction[104])

    # Logic:
    #1. Determine the number of sequence pairs.
        # This calculates the total number of possible input-output sequence pairs based on the input and output sequence length. Since this varies based on the objective, this is assigned a value within each objective sub-function and passed to this sub-function when called.
    #2. Generate the sequence pairs.
        # This section loops through the data to create the aforementioned sequence pairs given input length, output length, and stride. It then adds another dimension to make it compatible with the LSTM input format (3D tensor)
    #3. Split the data into a testing/validation set, and a training set using Sci-Kit Learns train_test_split
# ========================
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
#%%=======================
# 2c. Build and Train Long, Short-Term Memory (LSTM) Model

    #This section builds and trains the LSTM model. The architecture is similar to the MLP model homework, except this utilizes LSTM layers instead of dense layers. An additional "Checkpoint" callback has been implemented to save the best performing model during training for future troubleshooting purposes.
# ========================
def build_lstm(input_sequence_length, output_sequence_length, input_train_validation, output_train_validation):

    inputs=tf.Input((input_sequence_length,1))
    lstm_1=tf.layers.LSTM(32, return_sequences=True)(inputs)
    lstm_2=tf.layers.LSTM(32, return_sequences=False)(lstm_1)
    dense1=tf.layers.Dense(64, activation='relu')(lstm_2)
    outputs=tf.layers.Dense(output_sequence_length)(dense1)

    lstm_model = tf.Model(inputs=inputs, outputs=outputs)

    lstm_model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError()])

    early_stop = tf.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True,
                                            min_delta=0.00001)

    checkpoint = tf.callbacks.ModelCheckpoint(f"lstmmodel_{input_sequence_length}_{output_sequence_length}.keras",
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')

    # Train model
    lstm_history = lstm_model.fit(input_train_validation, output_train_validation,
                   epochs=1000,
                   verbose=1,
                   callbacks=[early_stop,checkpoint],
                   validation_data=.2)#use a value. Ref HW 1

    return lstm_model, lstm_history

#%%=======================
# 2d. Load Model

    #This section is used to troubleshoot the program without having to retrain on every run. By utilizing the checkpoint callback ('checkpoint' found in subsection 2c), we can save the best model and load it later.
# ========================
def load_lstm(input_sequence_length, output_sequence_length):

    lstm_model = tf.models.load_model(f"lstmmodel_{input_sequence_length}_{output_sequence_length}.keras")

    return lstm_model
#%% ====== SECTION 3: Section Objectives and Plotting ======

# Objective 1: Develop a baseline model with an input sequence length of 16.33 ms (50 data points) and an output sequence length of 16.33 ms (50 data points). Plot the model-predicted signal vs. the true signal.

# Objective 2: Vary the input and output sequence lengths to evaluate their effect on the error of the model predictions.

#%%=======================
# 3a. Objective 1

    #Since this objective involves comparing the performance of the model between a predicted future sequence and a true future sequence, a random sequence is predicted (In this case, at the 50th index value of the vapor fraction data) and plotted against the true values.
# ========================
def objective_1():

    input_sequence_length = 50
    output_sequence_length = 50
    stride = 1

    input_train_validation, input_test, output_train_validation, output_test = data_prep(input_sequence_length, output_sequence_length, stride)

    lstm_model, lstm_history = build_lstm(input_sequence_length, output_sequence_length, input_train_validation, output_train_validation)
    #lstm_model = load_lstm(input_sequence_length, output_sequence_length)

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

#%%=======================
# 3b. Objective 2

    #This objective loops through with multiple input/output sequence lengths and finds the mean squared error at each value. It then creates a heatmap plot of the MSE to determine which variable has the largest effect on performance.
# ========================
def objective_2():

    input_sequence_length = [25, 50, 75, 100, 125]
    output_sequence_length = [25, 50, 75, 100, 125]
    stride = 10

    error_results = {}

    for i in input_sequence_length:
        for j in output_sequence_length:

            input_train_validation, input_test, output_train_validation, output_test = data_prep(i, j, stride)

            lstm_model, lstm_history = build_lstm(i, j, input_train_validation, output_train_validation)
            #lstm_model = load_lstm(i, j)

            lstm_mean_squared_error = min(lstm_history.history['mean_squared_error'])

            error_results[i,j] = lstm_mean_squared_error

    #Comment on how to build heatmap and/or figures
    mse_grid = np.zeros((5, 5))
    for idx, i in enumerate(input_sequence_length):
        for jdx, j in enumerate(output_sequence_length):
            mse_grid[idx, jdx] = error_results[i, j]

    plt.figure(figsize=(8, 6))
    sns.heatmap(mse_grid,
                annot=True,
                cmap='YlGnBu',
                xticklabels=output_sequence_length,
                yticklabels=input_sequence_length)

    plt.title('Mean Squared Error Heatmap')
    plt.xlabel('Output Sequence Length')
    plt.ylabel('Input Sequence Length')
    plt.show()

#%% ====== SECTION 4: Main Function Call ======
main()