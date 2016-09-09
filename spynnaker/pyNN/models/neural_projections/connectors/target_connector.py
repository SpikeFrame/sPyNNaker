import numpy
from pyNN.random import RandomDistribution
from spynnaker.pyNN.models.neural_projections.connectors.abstract_connector \
    import AbstractConnector


class TargetConnector(AbstractConnector):
    """
    Where the pre- and postsynaptic populations have the same size, connect
    cell i in the presynaptic pynn_population.py to cell i in the postsynaptic
    pynn_population.py for all i.
    """

    def __init__(
            self, weights=0.0, source=None, space=None, safe=True, delays=1, verbose=False):
        """
        :param weights:
            may either be a float, a !RandomDistribution object, a list/
            1D array with at least as many items as connections to be
            created. Units nA.
        :param delays:
            as `weights`. If `None`, all synaptic delays will be set
            to the global minimum delay.

        """
        AbstractConnector.__init__(self, safe, space, verbose)

        if source == 'target':      # connection from target
            self._weights = 1       # to output layer

        elif source == 'output':    # connection from output neuron
            self._weights = 2       # back onto itself

        elif source == 'targetPre': # connection from target
            self._weights = 3       # to previous layer

        elif source == 'outputPre': # connection from output neuron
            self._weights = 4       # to previous layer

        elif source == 'hidden':    # connection from hidden neuron
            self._weights = 5       # back onto itself

        elif source == 'start':     # connection from target to start learning
            self._weights = 6       # need 2: 1 previous layer, 1 output layer

        elif source == 'stop':      # connection from target to stop learning
            self._weights = 7       # need 2: 1 previous layer, 1 output layer

        elif source == 'stopRegion':     # connection ending target range
            self._weights = 8            # to output layer

        elif source == 'stopRegionPre':  # connection ending target range
            self._weights = 9            # to previous layer

        elif source == 'startRegion':    # connection starting target range
            self._weights = 10           # to output layer

        elif source == 'startRegionPre': # connection starting target range
            self._weights = 11           # to previous layer

        else:
            print "\nFor the TargetConnector, we need initialized:" 
            print "source='target', source='output', source='targetPre' or source='outputPre'\n."
            import sys
            sys.exit()

        self._delays = 1

        self._check_parameters(self._weights, self._delays)

        # print 'self._weights:', self._weights

    def get_delay_maximum(self):
        return 1

    def get_delay_variance(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        if max_lo_atom > min_hi_atom:
            return 0
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        return self._get_delay_variance(self._delays, [connection_slice])

    def get_n_connections_from_pre_vertex_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            min_delay=None, max_delay=None):
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        if min_hi_atom < max_lo_atom:
            return 0
        return 1

    def get_n_connections_to_post_vertex_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        if min_hi_atom < max_lo_atom:
            return 0
        return 1

    def get_weight_mean(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        n_connections = (min_hi_atom - max_lo_atom) + 1
        if n_connections <= 0:
            return 0
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        return self._get_weight_mean(self._weights, [connection_slice])

    def get_weight_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        n_connections = (min_hi_atom - max_lo_atom) + 1
        if n_connections <= 0:
            return 0
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        return self._get_weight_maximum(
            self._weights, n_connections, [connection_slice])

    def get_weight_variance(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        if max_lo_atom > min_hi_atom:
            return 0
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        return self._get_weight_variance(self._weights, [connection_slice])

    def generate_on_machine(self):
        return (
            not self._generate_lists_on_host(self._weights) and
            not self._generate_lists_on_host(self._delays))

    def create_synaptic_block(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            synapse_type):
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        n_connections = max((0, (min_hi_atom - max_lo_atom) + 1))
        if n_connections <= 0:
            return numpy.zeros(0, dtype=AbstractConnector.NUMPY_SYNAPSES_DTYPE)
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        block = numpy.zeros(
            n_connections, dtype=AbstractConnector.NUMPY_SYNAPSES_DTYPE)
        block["source"] = numpy.arange(max_lo_atom, min_hi_atom + 1)
        block["target"] = numpy.arange(max_lo_atom, min_hi_atom + 1)
        block["weight"] = self._generate_weights(
            self._weights, n_connections, [connection_slice])
        block["delay"] = self._generate_delays(
            self._delays, n_connections, [connection_slice])
        block["synapse_type"] = synapse_type
        return block
