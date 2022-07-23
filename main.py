import plac

from SNN import SNN
from Parameters import Parameters

if __name__ == '__main__':
    initial_parameters = plac.call(Parameters)
    snn = SNN(initial_parameters)
    snn.run()