
from spynnaker.pyNN.models.neuron.plasticity.stdp.common \
    import plasticity_helpers
from spynnaker.pyNN.models.neuron.plasticity.stdp.timing_dependence\
    .abstract_timing_dependence import AbstractTimingDependence
from spynnaker.pyNN.models.neuron.plasticity.stdp.synapse_structure\
    .synapse_structure_weight_target import SynapseStructureWeightTarget


import logging
logger = logging.getLogger(__name__)

LOOKUP_TAU_PLUS_SIZE = 256
LOOKUP_TAU_PLUS_SHIFT = 0


class TimingDependenceSpikeTarget(AbstractTimingDependence):

    def __init__(self, tau_mem=20.0, nearest=False):
        AbstractTimingDependence.__init__(self)
        self._tau_plus = tau_mem
        self._nearest = nearest

        self._synapse_structure = SynapseStructureWeightTarget()

        # provenance data
        self._tau_plus_last_entry = None

    @property
    def tau_plus(self):
        return self._tau_plus

    @property
    def nearest(self):
        return self._nearest

    def is_same_as(self, timing_dependence):
        if not isinstance(timing_dependence, TimingDependenceSpikeTarget):
            return False
        return (
            (self._tau_plus == timing_dependence._tau_plus) and
            (self._nearest == timing_dependence._nearest))

    @property
    def vertex_executable_suffix(self):
        return "nearest_pair" if self._nearest else "pair"

    @property
    def pre_trace_n_bytes(self):

        # Pair rule requires no pre-synaptic trace when only the nearest
        # Neighbours are considered and, a single 16-bit R1 trace
        return 0 # if self._nearest else 2

    def get_parameters_sdram_usage_in_bytes(self):
        # 2*16bit for the two accumulators plus
        # 16bit weight * lookup tables
        return (2*2) + (2 * LOOKUP_TAU_PLUS_SIZE)

    @property
    def n_weight_terms(self):
        return 1

    def write_parameters(self, spec, machine_time_step, weight_scales):

        # Check timestep is valid
        if machine_time_step != 1000:
            raise NotImplementedError(
                "STDP LUT generation currently only supports 1ms timesteps")

        # Write lookup tables
        self._tau_plus_last_entry = plasticity_helpers.write_exp_lut(
            spec, self._tau_plus, LOOKUP_TAU_PLUS_SIZE,
            LOOKUP_TAU_PLUS_SHIFT)

    @property
    def synaptic_structure(self):
        return self._synapse_structure

    def get_provenance_data(self, pre_population_label, post_population_label):
        prov_data = list()
        prov_data.append(plasticity_helpers.get_lut_provenance(
            pre_population_label, post_population_label, "SpikePairRule",
            "tau_plus_last_entry", "tau_plus", self._tau_plus_last_entry))
        return prov_data
