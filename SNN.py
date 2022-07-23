import os
import pathlib
import time
import imageio
import imageio.v3 as iio
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import plac

class Parameters:
    # Script Settings
    mode = "train"
    weights_path = "weights.csv"
    labels_path = "labels.csv"
    image_inference_path = "image.png"
    debugging_interval = 1
    checkpoint_interval = 1
    train_dataset_path = "MNIST/training/"
    test_dataset_path = "MNIST/testing/"
    preprocessing_data = True
    training_images_amount = 100
    test_images_amount = 10
    use_tf_dataset: bool = True
    plotting_potentials = False
    visualize_weights = False
    testing = True

    # Simulation Parameters
    image_train_time = 350  # Training time for every image
    past_window = -5
    epochs = 3

    # Input Parameters
    image_size = (28, 28)
    resting_potential = -70
    layer1_size = image_size[0] * image_size[1]  # Number of neurons in first layer
    layer2_size = 100  # Number of neurons in second layer

    # Neuron Parameters
    inhibitory_potential = -100
    spike_threshold = -55
    hyperpolarization_potential = -90
    spike_drop_rate = 0.8
    threshold_drop_rate = 0.4
    min_weight = 0.00001
    max_weight = 1.0

    # STDP Parameters
    STDP_offset = 0  # STDP Offset
    sigma = 0.01  # STDP Learning Rate
    A_plus = 0.8  # Scaling Factor for Training Purposes - Decreasing Synpatic Weights
    A_minus = 0.3  # Scaling Factor for Training Purposes - Increasing Synpatic Weights
    tau_plus = 5  # STDP Time Constant in "μs" - Can be changed as Hyperparameter(?)
    tau_minus = 5  # STDP Time Constant in "μs" - Can be changed as Hyperparameter(?)
    mu = 0.9  # Exponential Factor

    # Receptive Field
    weight1 = 0.625
    weight2 = 0.125
    weight3 = -0.125
    weight4 = -.5

    # Spike Train Coding
    min_frequency = 1
    max_frequency = 50

    @plac.opt('mode', abbrev="mode",help="What Mode to use?", choices=['train', 'test', 'inference'])
    @plac.opt('weights_path', abbrev="weights_path",help="Path to weights file", type=pathlib.Path)
    @plac.opt('labels_path', abbrev="labels_path",help="Path to labels file", type=pathlib.Path)
    @plac.opt('image_inference_path', abbrev="image_inference_path", help="Path to image to infer", type=pathlib.Path)
    @plac.opt('debugging_interval', abbrev="debugging_interval", help="Script Setting", type=int)
    @plac.opt('checkpoint_interval', abbrev='checkpoint_interval', help="Script Setting", type=int)
    @plac.opt('train_dataset_path', abbrev='train_dataset_path', help="Script Setting", type=pathlib.Path)
    @plac.opt('test_dataset_path', abbrev='test_dataset_path', help="Script Setting", type=pathlib.Path)
    @plac.opt('preprocessing_data', abbrev='preprocessing_data', help="Script Setting", type=bool)
    @plac.opt('training_images_amount', abbrev='training_images_amount', help="Script Setting", type=int)
    @plac.opt('test_images_amount', abbrev='test_images_amount', help="Script Setting", type=int)
    @plac.opt('use_tf_dataset', abbrev='use_tf_dataset', help="Script Setting", type=bool)
    @plac.opt('plotting_potentials', abbrev='plotting_potentials', help="Script Setting", type=bool)
    @plac.opt('visualize_weights', abbrev='visualize_weights', help="Script Setting", type=bool)
    @plac.opt('testing', abbrev='testing', help="Script Setting", type=bool)
    @plac.opt('image_train_time', abbrev='image_train_time', help="Simulation Parameter", type=int)
    @plac.opt('past_window', abbrev='past_window', help="Simulation Parameter", type=int)
    @plac.opt('epochs', abbrev='epochs', help="Simulation Parameter", type=int)
    @plac.opt('image_size', abbrev='image_size', help="Input Parameter", type=tuple)
    @plac.opt('resting_potential', abbrev='resting_potential', help="Simulation Parameter", type=int)
    @plac.opt('layer1_size', abbrev='layer1_size', help="Simulation Parameter", type=int)
    @plac.opt('layer2_size', abbrev='layer2_size', help="Simulation Parameter", type=int)
    @plac.opt('inhibitory_potential', abbrev='inhibitory_potential', help="Neuron Parameter", type=int)
    @plac.opt('spike_threshold', abbrev='spike_threshold', help="Neuron Parameter", type=int)
    @plac.opt('hyperpolarization_potential', abbrev='hyperpolarization_potential', help="Neuron Parameter", type=int)
    @plac.opt('spike_drop_rate', abbrev='spike_drop_rate', help="Neuron Parameter", type=float)
    @plac.opt('threshold_drop_rate', abbrev='threshold_drop_rate', help="Neuron Parameter", type=float)
    @plac.opt('min_weight', abbrev='min_weight', help="Neuron Parameter", type=float)
    @plac.opt('max_weight', abbrev='max_weight', help="Neuron Parameter", type=float)
    @plac.opt('STDP_offset', abbrev='STDP_offset', help="STDP Parameter", type=float)
    @plac.opt('sigma', abbrev='sigma', help="STDP Parameter", type=float)
    @plac.opt('A_plus', abbrev='A_plus', help="STDP Parameter", type=float)
    @plac.opt('A_minus', abbrev='A_minus', help="STDP Parameter", type=float)
    @plac.opt('tau_plus', abbrev='tau_plus', help="STDP Parameter", type=float)
    @plac.opt('tau_minus', abbrev='tau_minus', help="STDP Parameter", type=float)
    @plac.opt('mu', abbrev='mu', help="STDP Parameter", type=float)
    @plac.opt('weight1', abbrev='weight1', help="Receptive Field Parameter", type=float)
    @plac.opt('weight2', abbrev='weight2', help="Receptive Field Parameter", type=float)
    @plac.opt('weight3', abbrev='weight3', help="Receptive Field Parameter", type=float)
    @plac.opt('weight4', abbrev='weight4', help="Receptive Field Parameter", type=float)
    @plac.opt('min_frequency', abbrev='min_frequency', help="Spike Train Coding Parameter", type=float)
    @plac.opt('max_frequency', abbrev='max_frequency', help="Spike Train Coding Parameter", type=float)
    def __init__(self, mode=mode, weights_path=weights_path, labels_path=labels_path, image_inference_path=image_inference_path, debugging_interval=debugging_interval, checkpoint_interval=checkpoint_interval, train_dataset_path=train_dataset_path, test_dataset_path=test_dataset_path, preprocessing_data=preprocessing_data, training_images_amount=training_images_amount, test_images_amount=test_images_amount, use_tf_dataset=use_tf_dataset, plotting_potentials=plotting_potentials, visualize_weights=visualize_weights, testing=testing, image_train_time=image_train_time, past_window=past_window, epochs=epochs, image_size=image_size, resting_potential=resting_potential, layer1_size=layer1_size, layer2_size=layer2_size, inhibitory_potential=inhibitory_potential, spike_threshold=spike_threshold, hyperpolarization_potential=hyperpolarization_potential, spike_drop_rate=spike_drop_rate, threshold_drop_rate=threshold_drop_rate, min_weight=min_weight, max_weight=max_weight, STDP_offset=STDP_offset, sigma=sigma, A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus, mu=mu, weight1=weight1, weight2=weight2, weight3=weight3, weight4=weight4, min_frequency=min_frequency, max_frequency=max_frequency):
        self.mode = mode
        self.weights_path = weights_path
        self.labels_path = labels_path
        self.image_inference_path = image_inference_path
        self.debugging_interval = debugging_interval
        self.checkpoint_interval = checkpoint_interval
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.preprocessing_data = preprocessing_data
        self.training_images_amount = training_images_amount
        self.test_images_amount = test_images_amount
        self.use_tf_dataset = use_tf_dataset
        self.plotting_potentials = plotting_potentials
        self.visualize_weights = visualize_weights
        self.testing = testing
        self.image_train_time = image_train_time
        self.past_window = past_window
        self.epochs = epochs
        self.image_size = image_size
        self.resting_potential = resting_potential
        self.layer1_size = image_size[0] * image_size[1]  # Number of neurons in first layer
        self.layer2_size = layer2_size
        self.inhibitory_potential = inhibitory_potential
        self.spike_threshold = spike_threshold
        self.hyperpolarization_potential = hyperpolarization_potential
        self.spike_drop_rate = spike_drop_rate
        self.threshold_drop_rate = threshold_drop_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.STDP_offset = STDP_offset
        self.sigma = sigma
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.mu = mu
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.weight4 = weight4
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency


class Neuron:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

        self.adaptive_spike_threshold = None
        self.refractory_period = None
        self.potential = None
        self.rest_until = None

        self.initial()

    def hyperpolarization(self, time_step):
        self.potential = self.parameters.hyperpolarization_potential
        self.rest_until = time_step + self.refractory_period

    def inhibit(self, time_step):
        self.potential = self.parameters.inhibitory_potential
        self.rest_until = time_step + self.refractory_period

    def initial(self):
        self.adaptive_spike_threshold = self.parameters.spike_threshold
        self.rest_until = -1
        self.refractory_period = 15  # (us)
        self.potential = self.parameters.resting_potential


class SNN:

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

        self.initial_timestamp = time.time()
        self.readable_initial_timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")


    def run(self):
        if self.parameters.mode == "train":
            self.train()

        elif self.parameters.mode == "test":
            pass
            raise NotImplemented
            # self.test()

        elif self.parameters.mode == "inference":
            prediction = self.inference(self.parameters.image_inference_path, self.parameters.weights_path, self.parameters.labels_path)
            print(f"Prediction: {prediction}")

    def encode_image_to_spike_train(self, image: np.ndarray):
        spike_trains = []

        for x_position in range(image.shape[0]):
            for y_position in range(image.shape[1]):

                pixel_value = image[x_position][y_position]

                spike_train = np.zeros(shape=(self.parameters.image_train_time + 1,))

                # Transfer pixel value to set frequency range(and some other stuff, which interp does...)
                frequency = np.interp(pixel_value, [np.min(image), np.max(image)], [self.parameters.min_frequency, self.parameters.max_frequency])

                spike_time_distance = math.ceil(self.parameters.image_train_time / frequency)
                next_spike_time = spike_time_distance

                if pixel_value > 0:
                    while next_spike_time < (self.parameters.image_train_time + 1):
                        # Add Spike to Spike Train
                        spike_train[int(next_spike_time)] = 1

                        # Calculate next spike
                        next_spike_time += spike_time_distance

                spike_trains.append(spike_train)

        return spike_trains

    def receptive_field(self, image: np.ndarray):
        image_size_x = image.shape[0]
        image_size_y = image.shape[1]

        # Receptive Field Kernel
        receptive_field = [
            [self.parameters.weight4, self.parameters.weight3, self.parameters.weight2, self.parameters.weight3, self.parameters.weight4],
            [self.parameters.weight3, self.parameters.weight2, self.parameters.weight1, self.parameters.weight2, self.parameters.weight3],
            [self.parameters.weight2, self.parameters.weight1, 1, self.parameters.weight1, self.parameters.weight2],
            [self.parameters.weight3, self.parameters.weight2, self.parameters.weight1, self.parameters.weight2, self.parameters.weight3],
            [self.parameters.weight4, self.parameters.weight3, self.parameters.weight2, self.parameters.weight3, self.parameters.weight4]]

        convoluted_image = np.zeros(shape=image.shape)

        window = [-2, -1, 0, 1, 2]
        x_offset = 2
        y_offset = 2

        # Apply Convolution with Receptive Field Kernel
        for x_image_index in range(image_size_x):
            for y_image_index in range(image_size_y):
                summation = 0
                for x_kernel_index in window:
                    for y_kernel_index in window:
                        if (x_image_index + x_kernel_index) >= 0 and (
                                x_image_index + x_kernel_index) <= image_size_x - 1 and (
                                y_image_index + y_kernel_index) >= 0 and (
                                y_image_index + y_kernel_index) <= image_size_y - 1:
                            summation = summation + (receptive_field[x_offset + x_kernel_index][y_offset + y_kernel_index] *
                                                     image[x_image_index + x_kernel_index][
                                                         y_image_index + y_kernel_index]) / 255
                convoluted_image[x_image_index][y_image_index] = summation
        return convoluted_image

    # STDP reinforcement learning curve
    def STDP_weighting_curve(self, delta_time: int):
        if delta_time > 0:
            return -self.parameters.A_plus * (np.exp(-float(delta_time) / self.parameters.tau_plus) - self.parameters.STDP_offset)
        if delta_time <= 0:
            return self.parameters.A_minus * (np.exp(float(delta_time) / self.parameters.tau_minus) - self.parameters.STDP_offset)

    # STDP weight update rule
    def update_synapse(self, synapse_weight, weight_factor):
        if weight_factor < 0:
            return synapse_weight + self.parameters.sigma * weight_factor * (synapse_weight - abs(self.parameters.min_weight)) ** self.parameters.mu
        elif weight_factor > 0:
            return synapse_weight + self.parameters.sigma * weight_factor * (self.parameters.max_weight - synapse_weight) ** self.parameters.mu

    def convert_weights_to_image(self, weights):
        weights = np.array(weights)
        weights = np.reshape(weights, self.parameters.image_size)
        image = np.zeros(self.parameters.image_size)
        for x_coordinate in range(self.parameters.image_size[0]):
            for y_coordinate in range(self.parameters.image_size[1]):
                image[x_coordinate][y_coordinate] = int(
                    np.interp(weights[x_coordinate][y_coordinate], [self.parameters.min_weight, self.parameters.max_weight], [0, 255]))
        return image

    def preprocess_data_tf_dataset(self, images, given_labels):
        spike_trains = []
        labels = []
        for image, label in zip(images, given_labels):
            spike_train = np.array(self.encode_image_to_spike_train(self.receptive_field(image)))
            spike_trains.append(spike_train)
            labels.append(label)
        return spike_trains, labels

    def preprocess_data_downloaded(self, dataset_path, number_of_pictures_per_class):
        number_of_pictures_per_class = int(number_of_pictures_per_class)
        spike_trains = []
        labels = []
        for index, folder in enumerate(os.listdir(dataset_path)):
            for image_path in os.listdir(dataset_path / folder)[:number_of_pictures_per_class]:
                image = iio.imread(dataset_path / folder / image_path)
                spike_train = np.array(self.encode_image_to_spike_train(self.receptive_field(image)))
                spike_trains.append(spike_train)
                labels.append(folder)
        return spike_trains, labels

    def train(self):
        print("Starting Training...")

        # Update initial timestamp
        self.initial_timestamp = time.time()
        self.readable_initial_timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")

        if self.parameters.plotting_potentials:
            potentials = []
            potential_thresholds = []

            for image_path in range(self.parameters.layer2_size):
                potentials.append([])
                potential_thresholds.append([])

        testing_accuracies = []

        time_of_learning = np.arange(1, self.parameters.image_train_time + 1, 1)

        output_layer = [Neuron(self.parameters) for i in range(self.parameters.layer2_size)]

        # Random Synapse Matrix	Initialization
        synapses = np.ones((self.parameters.layer2_size, self.parameters.layer1_size))  # Alternative: np.random.uniform(low=0.95, high=1.0, size=(self.parameters.layer2_size, self.parameters.layer1_size))  # Working Option: np.ones((layer2_size , layer1_size))
        self.parameters.max_weight = np.max(synapses)

        synapse_memory = np.zeros((self.parameters.layer2_size, self.parameters.layer1_size))

        # Creating Mapping Neurons which contains the Number they have learned
        neuron_labels_lookup = np.repeat(-1, self.parameters.layer2_size)

        # Loading Dataset
        if self.parameters.use_tf_dataset:
            (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
            X_train, Y_train = X_train[:self.parameters.training_images_amount], Y_train[:self.parameters.training_images_amount]
            X_test, Y_test = X_test[:self.parameters.test_images_amount], Y_test[:self.parameters.test_images_amount]

            if self.parameters.preprocessing_data:
                print("Preprocessing data...")
                X_train, Y_train = self.preprocess_data_tf_dataset(X_train, Y_train)
                X_test, Y_test = self.preprocess_data_tf_dataset(X_test, Y_test)
                print("Preprocessing finised.")
        else:
            X_train_generator, Y_train_generator = lambda: self.downloaded_dataset_generator_X(self.parameters.train_dataset_path, self.parameters.training_images_amount / 10), lambda: self.downloaded_dataset_generator_Y(self.parameters.train_dataset_path, self.parameters.training_images_amount / 10)
            X_test_generator, Y_test_generator = lambda: self.downloaded_dataset_generator_X(self.parameters.test_dataset_path, self.parameters.test_images_amount / 10), lambda: self.downloaded_dataset_generator_Y(self.parameters.test_dataset_path, self.parameters.test_images_amount / 10)

            if self.parameters.preprocessing_data:
                print("Preprocessing data...")
                X_train, Y_train = self.preprocess_data_downloaded(self.parameters.train_dataset_path, self.parameters.training_images_amount / 10)
                X_test, Y_test = self.preprocess_data_downloaded(self.parameters.test_dataset_path, self.parameters.test_images_amount / 10)
                print("Preprocessing finised.")


        for epoch in range(self.parameters.epochs):

            if not self.parameters.use_tf_dataset and not self.parameters.preprocessing_data:
                X_train, Y_train = X_train_generator(), Y_train_generator()
                X_test, Y_test = X_test_generator(), Y_test_generator()

            for image, label in zip(X_train, Y_train):
                time_start = time.time()

                if self.parameters.preprocessing_data:
                    spike_train = image
                else:
                    # Convolving image with receptive field and encoding to generate spike train
                    spike_train = np.array(self.encode_image_to_spike_train(self.receptive_field(image)))

                # Local variables
                winner_index = None
                count_spikes = np.zeros(self.parameters.layer2_size)
                current_potentials = np.zeros(self.parameters.layer2_size)

                synapse_memory = np.zeros((self.parameters.layer2_size, self.parameters.layer1_size))

                # Leaky integrate and fire neuron dynamics
                for time_step in time_of_learning:

                    for neuron_index, neuron in enumerate(output_layer):
                        self.calculate_potentials_and_adapt_thresholds(current_potentials, neuron, neuron_index, spike_train, synapses, time_step)

                        if self.parameters.plotting_potentials:
                            potentials[neuron_index].append(neuron.potential)  # Only for plotting: Changing potential overtime
                            potential_thresholds[neuron_index].append(neuron.adaptive_spike_threshold)  # Only for plotting: Changing threshold overtime

                    winner_index, winner_neuron = self.get_winner_neuron(current_potentials, output_layer)

                    # Check if Winner doesn't spike(reaches its own adaptive spike threshold?)
                    if current_potentials[winner_index] < winner_neuron.adaptive_spike_threshold:
                        continue  # Go to next time step

                    count_spikes[winner_index] += 1

                    winner_neuron.hyperpolarization(time_step)
                    winner_neuron.adaptive_spike_threshold += 1  # Adaptive Membrane / Homoeostasis: Increasing the threshold of the neuron

                    for layer1_index in range(self.parameters.layer1_size):
                        self.find_and_strengthen_contributing_synapses(layer1_index, spike_train, synapses, time_step, winner_index, synapse_memory)
                        self.find_and_weaken_not_contributing_synapses(layer1_index, synapse_memory, synapses, winner_index)

                    self.inihibit_looser_neurons(count_spikes, output_layer, time_step, winner_index)

                self.reset_neurons(output_layer)

                # Assigning Label to Winner Neuron
                neuron_labels_lookup[winner_index] = int(label)

            # TODO PROBABLY WRONG AND DANGEROUS OR SUPER IMPORTANT
            """for layer2_index in range(self.parameters.layer2_size):
                if neuron_labels_lookup[layer2_index] == -1:
                    for layer1_index in range(self.parameters.layer1_size):
                        synapses[layer2_index][layer1_index] = 0
"""
            if self.parameters.testing:
                testing_accuracy = self.test(synapses, neuron_labels_lookup, (X_test, Y_test))
                testing_accuracies.append(testing_accuracy)

            if epoch % self.parameters.debugging_interval == 0:
                self.debug_training_state(epoch, count_spikes, testing_accuracies)

            if epoch % self.parameters.checkpoint_interval == 0 and self.parameters.checkpoint_interval != 0:
                self.save_checkpoint(epoch, synapses, neuron_labels_lookup)

        # Final Checkpoint
        self.save_checkpoint("Final", synapses, neuron_labels_lookup)

        if self.parameters.plotting_potentials:
            self.plot_potentials_over_time(potential_thresholds, potentials)

        print("Finished Training. Saved Weights and Labels.")

    def save_checkpoint(self, epoch, synapses, neuron_labels_lookup):

        weights_path = f"Checkpoints/{self.readable_initial_timestamp}/Epoch_{epoch}/weights.csv"
        labels_path = f"Checkpoints/{self.readable_initial_timestamp}/Epoch_{epoch}/labels.csv"
        visualized_weights_path = f"Visualized_Weights/{self.readable_initial_timestamp}/Epoch_{epoch}/"

        pathlib.Path(weights_path[:weights_path.rindex(os.path.sep)]).mkdir(parents=True, exist_ok=True)
        pathlib.Path(labels_path[:labels_path.rindex(os.path.sep)]).mkdir(parents=True, exist_ok=True)
        pathlib.Path(visualized_weights_path[:visualized_weights_path.rindex(os.path.sep)]).mkdir(parents=True, exist_ok=True)

        np.savetxt(weights_path, synapses, delimiter=",")
        np.savetxt(labels_path, neuron_labels_lookup, delimiter=',')

        if self.parameters.visualize_weights:
            self.visualize_synapse_weights_and_save(neuron_labels_lookup, synapses, visualized_weights_path)


    def test(self, synapses, neuron_labels_lookup, dataset):
        X_test, Y_test = dataset
        predictions = []
        actual_labels = []
        for image, label in zip(X_test, Y_test):
            if self.parameters.preprocessing_data:
                spike_train = image
            else:
                spike_train = np.array(self.encode_image_to_spike_train(self.receptive_field(image)))
            prediction = self.infer(spike_train, synapses, neuron_labels_lookup)
            predictions.append(prediction)
            actual_labels.append(label)

        # Calculate Accuracy
        correct_predictions = 0
        for i in range(len(predictions)):
            if int(predictions[i]) == int(actual_labels[i]):
                correct_predictions += 1
        accuracy = correct_predictions / len(predictions)

        return accuracy

    def inference(self, image_path, synapse_weights_path=None, labels_matrix_path=None):
        if synapse_weights_path is None:
            synapse_weights_path = self.parameters.weights_path
        if labels_matrix_path is None:
            labels_matrix_path = self.parameters.labels_path
        image = iio.imread(image_path)
        spike_train = np.array(self.encode_image_to_spike_train(self.receptive_field(image)))
        synapses = np.loadtxt(synapse_weights_path, delimiter=",")
        neuron_labels_lookup = np.loadtxt(labels_matrix_path, delimiter=',')

        prediction = self.infer(spike_train, synapses, neuron_labels_lookup)

        return prediction

    def infer(self, spike_train, synapses, neuron_labels_lookup):
        synapses = synapses.copy()
        neuron_labels_lookup = neuron_labels_lookup.copy()

        # time series
        time_of_learning = np.arange(1, self.parameters.image_train_time + 1, 1)
        count_spikes = np.zeros((self.parameters.layer2_size, 1))
        output_layer = [Neuron(self.parameters) for i in range(self.parameters.layer2_size)]
        # flag for lateral inhibition
        current_potentials = np.zeros(self.parameters.layer2_size)
        winner_index = None
        for time_step in time_of_learning:
            for layer2_index, layer2_neuron in enumerate(output_layer):
                self.calculate_potentials_and_adapt_thresholds(current_potentials, layer2_neuron, layer2_index, spike_train, synapses, time_step)

            winner_index, winner_neuron = self.get_winner_neuron(current_potentials, output_layer)

            # Check if Winner doesn't spike(reaches its own adaptive spike threshold?)
            if current_potentials[winner_index] < winner_neuron.adaptive_spike_threshold:
                continue  # Go to next time step

            count_spikes[winner_index] += 1
            winner_neuron.hyperpolarization(time_step)
            winner_neuron.adaptive_spike_threshold += 1  #  Adaptive Membrane / Homoeostasis: Increasing the threshold of the neuron #TODO PLUS OR MINUS?!?

            self.inihibit_looser_neurons(count_spikes, output_layer, time_step, winner_index)
        prediction = neuron_labels_lookup[np.argmax(count_spikes)]
        return prediction

    def plot_potentials_over_time(self, potential_thresholds, potentials):
        # Plotting
        spaced_potentials = np.arange(0, len(potentials[0]), 1)
        for i in range(self.parameters.layer2_size):
            axes = plt.gca()
            plt.plot(spaced_potentials, potential_thresholds[i], 'r')
            plt.plot(spaced_potentials, potentials[i])
            plt.show()

    def visualize_synapse_weights_and_save(self, label_neuron, synapses, save_path):
        for layer2_index in range(self.parameters.layer2_size):
            if label_neuron[layer2_index] == -1:
                for layer1_index in range(self.parameters.layer1_size):
                    synapses[layer2_index][layer1_index] = 0
            image = self.convert_weights_to_image(synapses[layer2_index])
            imageio.imwrite(save_path / f'Neuron_{layer2_index}.png', image.astype(np.uint8))

    def debug_training_state(self, epoch, count_spikes, testing_accuracies):
        print(f"Epoch: {epoch} Testing Accuracy: {round(testing_accuracies[-1], 2) if testing_accuracies else 'None'} Time passed: {round(time.time() - self.initial_timestamp, 2)} seconds")

        # To write intermediate synapses for neurons
        # for p in range(layer2_size):
        #	reconst_weights(synapse[p],str(p)+"_epoch_"+str(k))

    def reset_neurons(self, output_layer):
        # Bring neuron potentials to rest
        for neuron_index, neuron in enumerate(output_layer):
            neuron.initial()

    def inihibit_looser_neurons(self, count_spikes, output_layer, time_step, winner_index):
        # Inhibit all LOOSERS
        for looser_neuron_index, looser_neuron in enumerate(output_layer):
            if looser_neuron_index != winner_index:
                if looser_neuron.potential > looser_neuron.adaptive_spike_threshold:
                    count_spikes[looser_neuron_index] += 1

                looser_neuron.inhibit(time_step)

    def find_and_strengthen_contributing_synapses(self, layer1_index, spike_train, synapses, time_step, winner_index, synapse_memory):
        """Part of STDP - Any synapse that contribute to the firing of a post-synaptic neuron should be increased. Depending on the timing of the pre- and postsynaptic spikes."""
        for past_time_step in range(0, self.parameters.past_window - 1, -1):  # if presynaptic spike came before postsynaptic spike
            if 0 <= time_step + past_time_step < self.parameters.image_train_time + 1:
                if spike_train[layer1_index][time_step + past_time_step] == 1:  # if presynaptic spike was in the tolerance window
                    synapses[winner_index][layer1_index] = self.update_synapse(synapses[winner_index][layer1_index], self.STDP_weighting_curve(past_time_step))  # strengthen weights
                    synapse_memory[winner_index][layer1_index] = 1
                    break
                # else:
                # synapse_memory[winner_index][layer1_index] = 0

    def find_and_weaken_not_contributing_synapses(self, layer1_index, synapse_memory, synapses, winner_index):
        if synapse_memory[winner_index][layer1_index] != 1:  # if presynaptic spike was not in the tolerance window, reduce weights of that synapse
            synapses[winner_index][layer1_index] = self.update_synapse(
                synapses[winner_index][layer1_index], self.STDP_weighting_curve(1))

    def get_winner_neuron(self, current_potentials, output_layer):
        winner_index = np.argmax(current_potentials)
        winner_neuron = output_layer[winner_index]
        return winner_index, winner_neuron

    def calculate_potentials_and_adapt_thresholds(self, current_potentials, neuron, neuron_index, spike_train, synapses, time_step):
        if neuron.rest_until < time_step:
            # Increase potential according to the sum of synapses inputs
            neuron.potential += np.dot(synapses[neuron_index], spike_train[:, time_step])

            if neuron.potential > self.parameters.resting_potential:
                neuron.potential -= self.parameters.spike_drop_rate

                if neuron.adaptive_spike_threshold > self.parameters.spike_threshold:
                    neuron.adaptive_spike_threshold -= self.parameters.threshold_drop_rate

            current_potentials[neuron_index] = neuron.potential

    def downloaded_dataset_generator_X(self, dataset_path, number_of_pictures_per_class):
        number_of_pictures_per_class = int(number_of_pictures_per_class)
        for index, folder in enumerate(os.listdir(dataset_path)):
            for image_path in os.listdir(dataset_path / folder)[:number_of_pictures_per_class]:
                image = iio.imread(dataset_path / folder / image_path)
                yield image

    def downloaded_dataset_generator_Y(self, dataset_path, number_of_pictures_per_class):
        number_of_pictures_per_class = int(number_of_pictures_per_class)
        for index, folder in enumerate(os.listdir(dataset_path)):
            for image_path in os.listdir(dataset_path / folder)[:number_of_pictures_per_class]:
                yield folder


if __name__ == '__main__':
    parameters = plac.call(Parameters)
    snn = SNN(parameters)
    snn.run()
