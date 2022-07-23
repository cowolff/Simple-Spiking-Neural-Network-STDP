from Parameters import Parameters

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
