import os
import time
import imageio
import numpy as np
import math


class Parameters:
    # Simulation Parameters
    image_train_time = 350  # Training time for every image
    past_window = -5
    epochs = 1

    # Input Parameters
    training_set_path = "./MNIST/training/"
    image_size = (28, 28)
    resting_potential = -70
    layer1_size = image_size[0] * image_size[1]  # Number of neurons in first layer
    layer2_size = 800  # Number of neurons in second layer

    # Neuron Parameters
    inhibitory_potential = -100
    spike_threshold = -55
    hyperpolarization_potential = -90
    spike_drop_rate = 0.8
    threshold_drop_rate = 0.4
    min_weight = 0.00001
    max_weight = 1.0

    # STDP Parameters
    STDP_offset = 0
    sigma = 0.01
    A_plus = 0.8
    A_minus = 0.8
    tau_plus = 5
    tau_minus = 5
    mu = 0.9

    min_frequency = 1
    max_frequency = 50


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

        weight1 = 0.625
        weight2 = 0.125
        weight3 = -0.125
        weight4 = -.5

        # Receptive Field Kernel
        receptive_field = [
            [weight4, weight3, weight2, weight3, weight4],
            [weight3, weight2, weight1, weight2, weight3],
            [weight2, weight1, 1, weight1, weight2],
            [weight3, weight2, weight1, weight2, weight3],
            [weight4, weight3, weight2, weight3, weight4]]

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
    def STDP_weighting_curve(self, time_step: int):
        if time_step > 0:
            return -self.parameters.A_plus * (np.exp(-float(time_step) / self.parameters.tau_plus) - self.parameters.STDP_offset)
        if time_step <= 0:
            return self.parameters.A_minus * (np.exp(float(time_step) / self.parameters.tau_minus) - self.parameters.STDP_offset)

    # STDP weight update rule
    def update_synapse(self, synapse_weight, weight_factor):
        if weight_factor < 0:
            return synapse_weight + self.parameters.sigma * weight_factor * (synapse_weight - abs(self.parameters.min_weight)) ** self.parameters.mu
        elif weight_factor > 0:
            return synapse_weight + self.parameters.sigma * weight_factor * (self.parameters.max_weight - synapse_weight) ** self.parameters.mu

    def convert_weights_to_image(self, weights, num):
        weights = np.array(weights)
        weights = np.reshape(weights, self.parameters.image_size)
        img = np.zeros(self.parameters.image_size)
        for x_coordinate in range(self.parameters.image_size[0]):
            for y_coordinate in range(self.parameters.image_size[1]):
                img[x_coordinate][y_coordinate] = int(
                    np.interp(weights[x_coordinate][y_coordinate], [self.parameters.min_weight, self.parameters.max_weight], [0, 255]))

        imageio.imwrite("visualized_weights/" + 'neuron_' + str(num) + '.png', img.astype(np.uint8))
        return img

    def training(self):
        potentials = []
        potential_thresholds = []
        for image_path in range(self.parameters.layer2_size):
            potentials.append([])
            potential_thresholds.append([])
        time_of_learning = np.arange(1, self.parameters.image_train_time + 1, 1)
        output_layer = []
        # Creating Second Layer
        for image_path in range(self.parameters.layer2_size):
            neuron = Neuron(self.parameters)
            neuron.initial()  # TODO In die __init__()
            output_layer.append(neuron)
        # Random synapse matrix	initialization
        synapses = np.ones((self.parameters.layer2_size, self.parameters.layer1_size))  # np.random.uniform(low=0.95, high=1.0, size=(self.parameters.layer2_size, self.parameters.layer1_size))  # np.ones((layer2_size , layer1_size))
        self.parameters.max_weight = np.max(synapses)
        synapse_memory = np.zeros((self.parameters.layer2_size, self.parameters.layer1_size))
        # Creating labels corresponding to neuron
        neuron_labels_lookup = np.repeat(-1, self.parameters.layer2_size)
        for epoch in range(self.parameters.epochs):
            for folder in next(os.walk('./MNIST/training/'))[1]:
                for image_path in os.listdir("./MNIST/training/" + folder + "/")[:80]:
                    time_start = time.time()

                    img = imageio.imread("./MNIST/training/" + folder + "/" + image_path)

                    # Convolving image with receptive field and encoding to generate spike train
                    spike_train = np.array(self.encode_image_to_spike_train(self.receptive_field(img)))

                    # Local variables
                    winner_index = None
                    count_spikes = np.zeros(self.parameters.layer2_size)
                    current_potentials = np.zeros(self.parameters.layer2_size)

                    # Leaky integrate and fire neuron dynamics
                    for time_step in time_of_learning:

                        for neuron_index, neuron in enumerate(output_layer):
                            self.calculate_potentials_and_adapt_thresholds(current_potentials, neuron, neuron_index, potential_thresholds, potentials, spike_train, synapses, time_step)

                        winner_index, winner_neuron = self.get_winner_neuron(current_potentials, output_layer)

                        # Check for spikes and update weights
                        if current_potentials[winner_index] < winner_neuron.adaptive_spike_threshold:
                            continue  # Go to next time step

                        count_spikes[winner_index] += 1

                        winner_neuron.hyperpolarization(time_step)
                        winner_neuron.adaptive_spike_threshold += 1  # Adaptive Membrane/Homoeostasis: Increasing the threshold of the neuron

                        for layer1_index in range(self.parameters.layer1_size):
                            self.find_and_strengthen_contributing_synapses(layer1_index, spike_train, synapse_memory, synapses, time_step, winner_index)
                            self.find_and_weaken_not_contributing_synapses(layer1_index, synapse_memory, synapses, winner_index)

                        self.inihibit_looser_neurons(count_spikes, output_layer, time_step, winner_index)

                    self.reset_neurons(output_layer)

                    neuron_labels_lookup[winner_index] = int(folder)

                    self.debug_training_state(count_spikes, time_start)

        """ print(count_spikes)
        # Plotting
        ttt = np.arange(0,len(potentials[0]),1)
        for i in range(layer2_size):
            axes = plt.gca()
            plt.plot(ttt, potential_thresholds[i], 'r' )
            plt.plot(ttt, potentials[i])
            plt.show() """
        # Reconstructing weights to analyze training
        self.visualize_synapse_weights(neuron_labels_lookup, synapses)
        np.savetxt("weights.csv", synapses, delimiter=",")
        np.savetxt("labels.csv", neuron_labels_lookup, delimiter=',')
        print("Finished Training. Saved Weights and Labels.")

    def visualize_synapse_weights(self, label_neuron, synapses):
        for layer2_index in range(self.parameters.layer2_size):
            if label_neuron[layer2_index] == -1:
                for layer1_index in range(self.parameters.layer1_size):
                    synapses[layer2_index][layer1_index] = 0
            self.convert_weights_to_image(synapses[layer2_index], str(layer2_index) + "_final")

    def debug_training_state(self, count_spikes, time_start):
        # print("Image: " + i + " Spike Count = ", count_spikes)
        print("Learning Neuron: ", np.argmax(count_spikes))
        print("Learning duration: ", time.time() - time_start)
        # to write intermediate synapses for neurons
        # for p in range(layer2_size):
        #	reconst_weights(synapse[p],str(p)+"_epoch_"+str(k))

    def reset_neurons(self, output_layer):
        # bring neuron potentials to rest
        for neuron_index, neuron in enumerate(output_layer):
            neuron.initial()

    def inihibit_looser_neurons(self, count_spikes, output_layer, time_step, winner_index):
        # Inhibit all LOOSERS
        for looser_neuron_index, looser_neuron in enumerate(output_layer):
            if looser_neuron_index != winner_index:
                if looser_neuron.potential > looser_neuron.adaptive_spike_threshold:
                    count_spikes[looser_neuron_index] += 1

                looser_neuron.inhibit(time_step)  # TODO So nothing happens for a few time steps afterwards?!?!?!?

    def find_and_strengthen_contributing_synapses(self, layer1_index, spike_train, synapse_memory, synapses, time_step, winner_index):
        """Part of STDP - Any synapse that contribute to the firing of a post-synaptic neuron should be increased. Depending on the timing of the pre- and postsynaptic spikes."""
        for past_time_step in range(0, self.parameters.past_window - 1, -1):  # if presynaptic spike came before postsynaptic spike
            if 0 <= time_step + past_time_step < self.parameters.image_train_time + 1:
                if spike_train[layer1_index][time_step + past_time_step] == 1:  # if presynaptic spike was in the tolerance window
                    synapses[winner_index][layer1_index] = self.update_synapse(synapses[winner_index][layer1_index], self.STDP_weighting_curve(past_time_step))  # strengthen weights
                    synapse_memory[winner_index][
                        layer1_index] = 1  # TODO Possible Reset necessary - somewhere?!?!?
                    break

    def find_and_weaken_not_contributing_synapses(self, layer1_index, synapse_memory, synapses, winner_index):
        if synapse_memory[winner_index][layer1_index] != 1:  # if presynaptic spike was not in the tolerance window, reduce weights of that synapse
            synapses[winner_index][layer1_index] = self.update_synapse(synapses[winner_index][layer1_index], self.STDP_weighting_curve(1))

    def get_winner_neuron(self, current_potentials, output_layer):
        winner_index = np.argmax(current_potentials)
        winner_neuron = output_layer[winner_index]
        return winner_index, winner_neuron

    def calculate_potentials_and_adapt_thresholds(self, current_potentials, neuron, neuron_index, potential_thresholds, potentials, spike_train, synapses, time_step):
        if neuron.rest_until < time_step:
            # Increase potential according to the sum of synapses inputs
            neuron.potential += np.dot(synapses[neuron_index], spike_train[:, time_step])

            if neuron.potential > self.parameters.resting_potential:
                neuron.potential -= self.parameters.spike_drop_rate

                if neuron.adaptive_spike_threshold > self.parameters.spike_threshold:
                    neuron.adaptive_spike_threshold -= self.parameters.threshold_drop_rate

            current_potentials[neuron_index] = neuron.potential
        potentials[neuron_index].append(
            neuron.potential)  # Only for plotting: Changing potential overtime
        potential_thresholds[neuron_index].append(
            neuron.adaptive_spike_threshold)  # Only for plotting: Changing threshold overtime


initial_parameters = Parameters()
snn = SNN(initial_parameters)
snn.training()
