from data_specification.data_specification_generator import \
    DataSpecificationGenerator
from data_specification.file_data_writer import FileDataWriter


from spynnaker.pyNN.utilities.conf import config
from spynnaker.pyNN.utilities import packet_conversions
from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.neural_properties.synaptic_manager import \
    SynapticManager
from spynnaker.pyNN.models.abstract_models.abstract_partitionable_vertex import \
    PartitionableVertex


import os
import math
import ctypes
import tempfile
import numpy
import logging

logger = logging.getLogger(__name__)


class PopulationManager(SynapticManager, PartitionableVertex):

    def __init__(self, record, binary, n_neurons, label, constraints):
        SynapticManager.__init__(self)
        PartitionableVertex.__init__(self, n_neurons, label, constraints)
        self._record = record
        self._record_v = False
        self._record_gsyn = False
        self._binary = binary

    def record_v(self):
        self._record_v = True

    def record_gsyn(self):
        self._record_gsyn = True

    def get_parameters(self, machine_time_step):
        raise NotImplementedError

    def reserve_memory_regions(self, spec, setup_sz, neuron_params_sz,
                               synapse_params_sz, row_len_trans_sz,
                               master_pop_table_sz, all_syn_block_sz,
                               spike_hist_buff_sz, potential_hist_buff_sz,
                               gsyn_hist_buff_sz, stdp_params_sz):
        """
        Reserve SDRAM space for memory areas:
        1) Area for information on what data to record
        2) Neuron parameter data (will be copied to DTCM by 'C'
           code at start-up)
        3) synapse parameter data (will be copied to DTCM)
        4) Synaptic row length look-up (copied to DTCM)
        5) Synaptic block look-up table. Translates the start address
           of each block of synapses (copied to DTCM)
        6) Synaptic row data (lives in SDRAM)
        7) Spike history
        8) Neuron potential history
        9) Gsyn value history
        """

        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserveMemRegion(region=constants.REGIONS.SYSTEM,
                              size=setup_sz,
                              label='setup')
        spec.reserveMemRegion(region=constants.REGIONS.NEURON_PARAMS,
                              size=neuron_params_sz,
                              label='NeuronParams')
        spec.reserveMemRegion(region=constants.REGIONS.SYNAPSE_PARAMS,
                              size=synapse_params_sz,
                              label='SynapseParams')
        spec.reserveMemRegion(region=constants.REGIONS.ROW_LEN_TRANSLATION,
                              size=row_len_trans_sz,
                              label='RowLenTable')
        spec.reserveMemRegion(region=constants.REGIONS.MASTER_POP_TABLE,
                              size=master_pop_table_sz,
                              label='MasterPopTable')
        spec.reserveMemRegion(region=constants.REGIONS.SYNAPTIC_MATRIX,
                              size=all_syn_block_sz,
                              label='SynBlocks')

        if self._record:
            spec.reserveMemRegion(region=constants.REGIONS.SPIKE_HISTORY,
                                  size=spike_hist_buff_sz,
                                  label='spikeHistBuffer',
                                  leaveUnfilled=True)
        if self._record_v:
            spec.reserveMemRegion(region=constants.REGIONS.POTENTIAL_HISTORY,
                                  size=potential_hist_buff_sz,
                                  label='potHistBuffer',
                                  leaveUnfilled=True)
        if self._record_gsyn:
            spec.reserveMemRegion(region=constants.REGIONS.GSYN_HISTORY,
                                  size=gsyn_hist_buff_sz,
                                  label='gsynHistBuffer',
                                  leaveUnfilled=True)
        if stdp_params_sz != 0:
            spec.reserveMemRegion(region=constants.REGIONS.STDP_PARAMS,
                                  size=stdp_params_sz,
                                  label='stdpParams')

    def write_setup_info(self, spec, spike_history_region_sz,
                         neuron_potential_region_sz, gsyn_region_sz,
                         timer_period):
        """
        Write information used to control the simulation and gathering of
        results.Currently, this means the flag word used to signal whether
        information on neuron firing and neuron potential is either stored
        locally in a buffer or passed out of the simulation for storage/display
         as the simulation proceeds.

        The format of the information is as follows:
        Word 0: Flags selecting data to be gathered during simulation.
            Bit 0: Record spike history
            Bit 1: Record neuron potential
            Bit 2: Record gsyn values
            Bit 3: Reserved
            Bit 4: Output spike history on-the-fly
            Bit 5: Output neuron potential
            Bit 6: Output spike rate
        """
        # What recording commands were set for the parent abstract_population.py?
        recording_info = 0
        if spike_history_region_sz > 0 and self._record:
            recording_info |= constants.RECORD_SPIKE_BIT
        if neuron_potential_region_sz > 0 and self._record_v:
            recording_info |= constants.RECORD_STATE_BIT
        if gsyn_region_sz > 0 and self._record_gsyn:
            recording_info |= constants.RECORD_GSYN_BIT
        recording_info |= 0xBEEF0000

        # Write this to the system region (to be picked up by the simulation):
        spec.switchWriteFocus(region=constants.REGIONS.SYSTEM)
        spec.write(data=timer_period)
        spec.write(data=recording_info)
        spec.write(data=spike_history_region_sz)
        spec.write(data=neuron_potential_region_sz)
        spec.write(data=gsyn_region_sz)

    def write_neuron_parameters(self, spec, machine_time_step, processor,
                                subvertex, ring_buffer_to_input_left_shift):
        spec.comment("\nWriting Neuron Parameters for {%d} "
                     "Neurons:\n".format(subvertex.n_atoms))

        # Set the focus to the memory region 2 (neuron parameters):
        spec.switchWriteFocus(region=constants.REGIONS.NEURON_PARAMS)

        # Write header info to the memory region:
        # Write Key info for this core:
        chip_x, chip_y, chip_p = processor.get_coordinates()
        population_identity = \
            packet_conversions.get_key_from_coords(chip_x, chip_y, chip_p)
        spec.write(data=population_identity)

        # Write the number of neurons in the block:
        spec.write(data=subvertex.n_atoms)

        # Write the number of parameters per neuron (struct size in words):
        params = self.get_parameters(machine_time_step)
        spec.write(data=len(params))

        # Write machine time step: (Integer, expressed in microseconds)
        spec.write(data=machine_time_step)

        # Write ring_buffer_to_input_left_shift
        spec.write(data=ring_buffer_to_input_left_shift)

        # TODO: NEEDS TO BE LOOKED AT PROPERLY
        # Create loop over number of neurons:
        for atom in range(0, subvertex.n_atoms):
            # Process the parameters
            for param in params:
                value = param.get_value()
                if hasattr(value, "__len__"):
                    if len(value) > 1:
                        value = value[atom]
                    else:
                        value = value[0]

                datatype = param.get_datatype()
                scale = param.get_scale()

                value = value * scale

                if datatype == 's1615':
                    value = spec.doubleToS1615(value)
                elif datatype == 'uint32':
                    value = ctypes.c_uint32(value).value

                spec.write(data=value, sizeof=datatype)
        # End the loop over the neurons:

    @staticmethod
    def get_ring_buffer_to_input_left_shift(subvertex):
        total_exc_weights = numpy.zeros(subvertex.n_atoms)
        total_inh_weights = numpy.zeros(subvertex.n_atoms)
        for subedge in subvertex.in_subedges:
            sublist = subedge.get_synapse_sublist()
            sublist.sum_weights(total_exc_weights, total_inh_weights)

        max_weight = max((max(total_exc_weights), max(total_inh_weights)))
        max_weight_log_2 = 0
        if max_weight > 0:
            max_weight_log_2 = math.log(max_weight, 2)

        # Currently, we can only cope with positive left shifts, so the minimum
        # scaling will be no shift i.e. a max weight of 0nA
        if max_weight_log_2 < 0:
            max_weight_log_2 = 0

        max_weight_power = int(math.ceil(max_weight_log_2))

        logger.debug("Max weight is {}, Max power is {}"
                     .format(max_weight, max_weight_power))

        # Actual shift is the max_weight_power - 1 for 16-bit fixed to s1615,
        # but we ignore the "-1" to allow a bit of overhead in the above
        # calculation in case a couple of extra spikes come in
        return max_weight_power

    def generate_data_spec(self, processor, subvertex, machine_time_step,
                           run_time):
        """
        Model-specific construction of the data blocks necessary to build a group
        of IF_curr_exp neurons resident on a single core.
        """
        # Create new DataSpec for this processor:
        x, y, p = processor.get_coordinates()
        hostname = processor.chip.machine.hostname
        has_binary_folder_set = \
            config.has_option("SpecGeneration", "Binary_folder")
        if not has_binary_folder_set:
            binary_folder = tempfile.gettempdir()
            config.set("SpecGeneration", "Binary_folder", binary_folder)
        else:
            binary_folder = config.get("SpecGeneration", "Binary_folder")

        binary_file_name = \
            binary_folder + os.sep + "{%s}_dataSpec_{%d}_{%d}_{%d}.dat"\
                                     .format(hostname, x, y, p)

        data_writer = FileDataWriter(binary_file_name)

        spec = DataSpecificationGenerator(data_writer)

        spec.comment("\n*** Spec for block of {%s} neurons ***\n"
                     .format(self.model_name))

        # Calculate the number of time steps
        no_machine_time_steps = int((run_time * 1000.0) / machine_time_step)

        x, y, p = processor.get_coordinates()

        # Calculate the size of the tables to be reserved in SDRAM:
        neuron_params_sz = self.get_neuron_params_size(subvertex.lo_atom,
                                                       subvertex.hi_atom)
        synapse_params_sz = self.get_synapse_parameter_size(subvertex.lo_atom,
                                                            subvertex.hi_atom)
        all_syn_block_sz = self.get_exact_synaptic_block_memory_size(subvertex)
        spike_hist_buff_sz = self.get_spike_buffer_size(subvertex.lo_atom,
                                                        subvertex.hi_atom,
                                                        no_machine_time_steps)
        potential_hist_buff_sz = self.get_v_buffer_size(subvertex.lo_atom,
                                                        subvertex.hi_atom,
                                                        no_machine_time_steps)
        gsyn_hist_buff_sz = self.get_g_syn_buffer_size(subvertex.lo_atom,
                                                       subvertex.hi_atom,
                                                       no_machine_time_steps)
        stdp_region_sz = self.get_stdp_parameter_size(subvertex.lo_atom,
                                                      subvertex.hi_atom,
                                                      self.in_edges)

        # Declare random number generators and distributions:
        self.write_random_distribution_declarations(spec)

        # Construct the data images needed for the Neuron:
        self.reserve_memory_regions(
            spec, constants.SETUP_SIZE, neuron_params_sz, synapse_params_sz,
            SynapticManager.ROW_LEN_TABLE_SIZE,
            SynapticManager.MASTER_POPULATION_TABLE_SIZE, all_syn_block_sz,
            spike_hist_buff_sz, potential_hist_buff_sz, gsyn_hist_buff_sz,
            stdp_region_sz)

        self.write_setup_info(spec, spike_hist_buff_sz, potential_hist_buff_sz,
                              gsyn_hist_buff_sz, machine_time_step)

        ring_buffer_shift = self.get_ring_buffer_to_input_left_shift(subvertex)
        weight_scale = self.get_weight_scale(ring_buffer_shift)
        logger.debug("Weight scale is {}".format(weight_scale))

        self.write_neuron_parameters(spec, machine_time_step, processor,
                                     subvertex, ring_buffer_shift)

        self.write_synapse_parameters(spec, machine_time_step, subvertex)

        self.write_stdp_parameters(spec, machine_time_step, subvertex,
                                   weight_scale, constants.REGIONS.STDP_PARAMS)

        self.write_row_length_translation_table(
            spec, constants.REGIONS.ROW_LEN_TRANSLATION)

        self.write_synaptic_matrix_and_master_population_table(
            spec, subvertex, all_syn_block_sz, weight_scale,
            constants.REGIONS.MASTER_POP_TABLE,
            constants.REGIONS.SYNAPTIC_MATRIX)

        for subedge in subvertex.in_subedges:
            subedge.free_sublist()

        # End the writing of this specification:
        spec.end_specification()
        data_writer.close()

        # No memory writes required for this Data Spec:
        memory_write_targets = list()
        simulation_time_in_ticks = constants.INFINITE_SIMULATION
        if run_time is not None:
            simulation_time_in_ticks = \
                int((run_time * 1000.0) / machine_time_step)
        user1_addr = \
            0xe5007000 + 128 * p + 116  # User1 location reserved for core p
        memory_write_targets.append({'address': user1_addr,
                                     'data': simulation_time_in_ticks})

         # Split binary name into title and extension
        binary_title, binary_extension = os.path.splitext(self._binary)

        # If we have an STDP mechanism, add it's executable suffic to title
        if self._stdp_mechanism is not None:
            binary_title = \
                binary_title + "_" + \
                self._stdp_mechanism.get_vertex_executable_suffix()

        # Rebuild executable name
        binary_name = os.path.join(config.get("SpecGeneration",
                                              "common_binary_folder"),
                                   binary_title + binary_extension)

        # Return list of target cores, executables, files to load and
        # memory writes to perform:
        return binary_name, list(), memory_write_targets