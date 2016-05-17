from spynnaker.pyNN.models.neuron.plasticity.stdp.synapse_structure.\
    abstract_synapse_structure import AbstractSynapseStructure
import numpy


class SynapseStructureWeightOnly(AbstractSynapseStructure):

    def __init__(self):
        AbstractSynapseStructure.__init__(self)

    def get_n_bytes_per_connection(self):
        return 6

    def get_synaptic_data(self, connections):
        # Get 16-bit weights
        weights = numpy.rint(numpy.abs(connections["weight"])).astype("uint16")
        
        # Create zeroed plastic plastic section large enough
        # for weights and 2 16-bit traces (48-bits)
        plastic_plastic = numpy.zeros((weights.shape[0] * 3), dtype="uint16")
        
        # Copy weights into every three half words (0, 3, 6 etc)
        plastic_plastic[0::3] = weights
        
        # Convert into format it seems to require and return
        return plastic_plastic.view(dtype="uint8").reshape((-1, 6))

    def read_synaptic_data(self, fp_size, pp_data):
        return numpy.concatenate([
            pp_data[i].view(dtype="uint16")[0:fp_size[i]]
            for i in range(len(pp_data))])
