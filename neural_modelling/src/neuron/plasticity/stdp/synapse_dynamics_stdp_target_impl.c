// Spinn_common includes
#include "static-assert.h"

// sPyNNaker neural modelling includes
#include "../../synapses.h"

// Plasticity common includes
#include "../common/maths.h"
#include "../common/post_events.h"

#include "weight_dependence/weight.h"
#include "timing_dependence/timing_target_pair_impl.h"
#include <string.h>
#include <debug.h>

// to start and end learning patterns
#include "timing_dependence/timing_target_pair_impl.h"

#ifdef SYNAPSE_BENCHMARK
 uint32_t num_plastic_pre_synaptic_events;
#endif  // SYNAPSE_BENCHMARK

//---------------------------------------
// Macros
//---------------------------------------
// The plastic control words used by Morrison synapses store an axonal delay
// in the upper 3 bits.
// Assuming a maximum of 16 delay slots, this is all that is required as:
//
// 1) Dendritic + Axonal <= 15
// 2) Dendritic >= Axonal
//
// Therefore:
//
// * Maximum value of dendritic delay is 15 (with axonal delay of 0)
//    - It requires 4 bits
// * Maximum value of axonal delay is 7 (with dendritic delay of 8)
//    - It requires 3 bits
//
// |        Axonal delay       |  Dendritic delay   |       Type        |      Index         |
// |---------------------------|--------------------|-------------------|--------------------|
// | SYNAPSE_AXONAL_DELAY_BITS | SYNAPSE_DELAY_BITS | SYNAPSE_TYPE_BITS | SYNAPSE_INDEX_BITS |
// |                           |                    |        SYNAPSE_TYPE_INDEX_BITS         |
// |---------------------------|--------------------|----------------------------------------|
#ifndef SYNAPSE_AXONAL_DELAY_BITS
#define SYNAPSE_AXONAL_DELAY_BITS 0 // Changed this from 3 to 0, because there's no target delay
#endif

#define SYNAPSE_AXONAL_DELAY_MASK ((1 << SYNAPSE_AXONAL_DELAY_BITS) - 1)

#define SYNAPSE_DELAY_TYPE_INDEX_BITS \
    (SYNAPSE_DELAY_BITS + SYNAPSE_TYPE_INDEX_BITS)

#if (SYNAPSE_DELAY_TYPE_INDEX_BITS + SYNAPSE_AXONAL_DELAY_BITS) > 16
#error "Not enough bits for axonal synaptic delay bits"
#endif

//---------------------------------------
// Structures
//---------------------------------------
typedef struct {
    uint32_t prev_time;
} pre_event_history_t;

post_event_history_t *post_event_history;

//---------------------------------------
// Synapse update loop
//---------------------------------------
static inline final_state_t _plasticity_update_synapse(
        uint32_t time,
        const uint32_t last_pre_time,
        const uint32_t delay_dendritic,
        const uint32_t delay_axonal, update_state_t current_state,
        const post_event_history_t *post_event_history) {
    use(delay_dendritic);

    // last output time of the hidden neuron
    static uint32_t hiddenOutTime = 0;

    // Apply axonal delay to time of last presynaptic spike
    const uint32_t delayed_last_pre_time = last_pre_time + delay_axonal;

    // Get the post-synaptic window of events to be processed
    const uint32_t window_begin_time = delayed_last_pre_time;
    const uint32_t window_end_time = time + delay_axonal;
    post_event_window_t post_window = post_events_get_window_delayed(
            post_event_history, window_begin_time, window_end_time);

    log_debug("\tPerforming deferred synapse update at time:%u", time);
    log_debug("\t\tbegin_time:%u, end_time:%u - prev_time:%u, num_events:%u",
        window_begin_time, window_end_time, post_window.prev_time,
        post_window.num_events);

//io_printf(IO_BUF,"%dms, inside _plasticity_update_synapse.    last_pre_time=%dms    post_window.num_events=%d    delayed_last_pre_time=%d\n", time, last_pre_time, post_window.num_events, delayed_last_pre_time);

    // Process events in post-synaptic window
    while (post_window.num_events > 0) {
        uint32_t delayed_post_time = *post_window.next_time;
        post_trace_t target_trace = *post_window.next_trace;

        log_debug("\t\tApplying post-synaptic event at delayed time:%u\n",
              delayed_post_time);

        // Get time of event relative to last pre-synaptic event
        // delayed_last_pre_time --> pre-synaptic spike time
        // delayed_post_time  -----> output spike time

//io_printf(IO_BUF,"%dms,   target_trace=%d    delayed_post_time=%dms    delayed_last_pre_time=%dms\n", time, target_trace, delayed_post_time, delayed_last_pre_time);


        // The synaptic signals that can be added to post-events:
        // target_trace, syn_signal = 1   # spike from target to output layer
        // target_trace, syn_signal = 2   # spike from output neuron back onto itself
        // target_trace, syn_signal = 3   # spike from target to previous layer
        // target_trace, syn_signal = 4   # spike from output neuron to previous layer
        // target_trace, syn_signal = 5   # spike from hidden neuron back onto itself
        // target_trace, syn_signal = 6   # starting learning
        // target_trace, syn_signal = 7   # ending learning
        switch(target_trace)
        {

            // a spike from Target to Output neuron (case 1)
            //        or Output to back onto itself (case 2)
            case 1:
            case 2:
//io_printf(IO_BUF,"%dms, delayed_post_time=%dms    delayed_last_pre_time=%dms\n", time, delayed_post_time, delayed_last_pre_time);
//io_printf(IO_BUF,"%dms, delayed_post_time=%dms    delayed_last_pre_time=%dms    old_accumulator:%d    ", time, delayed_post_time, delayed_last_pre_time, current_state.accumulator);
if ((delayed_post_time - delayed_last_pre_time) < 513)
{
                // last output spike must be after its presynaptic spike time
                if (delayed_post_time > delayed_last_pre_time) 
                {
                    current_state = timing_apply_post_spike( 
                                (delayed_post_time - delayed_last_pre_time),
                                                target_trace, 
                                                current_state);
                }
}
//io_printf(IO_BUF,"new_accumulator%d\n", current_state.accumulator);

                break;

            // a spike from Target to Hidden neuron (case 3)
            // or a spike from Output to Hidden neuron (case 4)
            case 3:
            case 4:
                if ((delayed_post_time - delayed_last_pre_time) < 513)
                {

                    //io_printf(IO_BUF,"%dms, delayed_post_time=%dms    delayed_last_pre_time=%dms    hiddenOutTime=%d\n", time, delayed_post_time, delayed_last_pre_time, hiddenOutTime);

                    // last hidden layer spike must be after its pre-syn spike
                    if (hiddenOutTime > delayed_last_pre_time)
                    {
                        // output spike time must be after the hidden spike time
                        if (delayed_post_time > hiddenOutTime)
                        {
                            //io_printf(IO_BUF,"%dms 2 conditions, delayed_post_time=%dms    delayed_last_pre_time=%dms\n", time, delayed_post_time, delayed_last_pre_time);
                            //io_printf(IO_BUF,"%dms: ", time);                        
                            current_state = timing_apply_post_spike( 
                                        (delayed_post_time - delayed_last_pre_time),
                                                                    target_trace, 
                                                                    current_state);
                        }

                    }
                }
                break;

            // a spike from Hidden neuron back onto itself (case 5)
            case 5:
                if (delayed_last_pre_time>0) 
                {
//io_printf(IO_BUF,"at time: %dms, Hidden neuron spike back onto itself.    delayed_post_time=%dms    delayed_last_pre_time=%dms\n", time, delayed_post_time, delayed_last_pre_time);
                hiddenOutTime = delayed_post_time; //time; // update hidden neuron spike output time
                }
                break;

            // starting learning
            case 6:
                //io_printf(IO_BUF,"at time: %dms, pattern begins.    delayed_post_time=%dms    delayed_last_pre_time=%dms\n", time, delayed_post_time, delayed_last_pre_time);
                current_state = pattern_begins(current_state);
                break;

            // ending learning
            case 7:     
                    //io_printf(IO_BUF,"%dms, pattern ends.    weight:%d + current_state.accumulator=%d =", time, current_state.weight_state.initial_weight, current_state.accumulator);
                    
//if (current_state.accumulator > 0)
//io_printf(IO_BUF,"%dms... pattern ends, increasing weight!!!\n", time);
                    current_state = pattern_ends(current_state);
                    //io_printf(IO_BUF," %d \n", current_state.weight_state.initial_weight);

                break;


    // (weight=8)  syn_signal = 8  # ending   target range to output   layer
    // (weight=9)  syn_signal = 9  # ending   target range to previous layer
    // (weight=10) syn_signal = 10 # starting target range to output   layer
    // (weight=11) syn_signal = 11 # starting target range to previous layer

            // ending target range to output layer
            case 8:
//io_printf(IO_BUF,"%dms, processing target spike at synapse!    delayed_post_time=%dms    delayed_last_pre_time=%dms\n", time, delayed_post_time, delayed_last_pre_time);


                // last output spike must be after its presynaptic spike time
                if (delayed_post_time > delayed_last_pre_time) 
                {
                                        // apply the post spike
                    current_state = timing_apply_post_spike( 
                                (delayed_post_time - delayed_last_pre_time),
                                1, 
                                current_state);
                }
                break;

            // ending target range to previous layer
            case 9:

                // last output spike must be after its presynaptic spike time
                if (delayed_post_time > delayed_last_pre_time) 
                {
                    // apply the post spike
                    current_state = timing_apply_post_spike( 
                                (delayed_post_time - delayed_last_pre_time),
                                3, 
                                current_state);
                }
                break;

            // learning pattern ends with no weight updates
            case 12:

//io_printf(IO_BUF,"%dms... pattern ends without learning! Resetting accumulator to 0!!!\n", time);

                // reset accumulator to baseline value
                current_state.accumulator = 0;

            break;

        }

        // Go onto next event
        post_window = post_events_next_delayed(post_window, delayed_post_time);
    }

    const uint32_t delayed_pre_time = time + delay_axonal;
    log_debug("\t\tApplying pre-synaptic event at time:%u last post time:%u\n",
              delayed_pre_time, post_window.prev_time);

    // Return final synaptic word and weight
    return synapse_structure_get_final_state(current_state);
}

//---------------------------------------
// Synaptic row plastic-region implementation
//---------------------------------------
static inline plastic_synapse_t* _plastic_synapses(
        address_t plastic_region_address) {
    const uint32_t pre_event_history_size_words =
        sizeof(pre_event_history_t) / sizeof(uint32_t);
    static_assert(pre_event_history_size_words * sizeof(uint32_t)
                  == sizeof(pre_event_history_t),
                  "Size of pre_event_history_t structure should be a multiple"
                  " of 32-bit words");

    return (plastic_synapse_t*)
        (&plastic_region_address[pre_event_history_size_words]);
}

//---------------------------------------
static inline pre_event_history_t *_plastic_event_history(
        address_t plastic_region_address) {
    return (pre_event_history_t*) (&plastic_region_address[0]);
}

void synapse_dynamics_print_plastic_synapses(
        address_t plastic_region_address, address_t fixed_region_address,
        uint32_t *ring_buffer_to_input_buffer_left_shifts) {
    use(plastic_region_address);
    use(fixed_region_address);
    use(ring_buffer_to_input_buffer_left_shifts);
#if LOG_LEVEL >= LOG_DEBUG

    // Extract separate arrays of weights (from plastic region),
    // Control words (from fixed region) and number of plastic synapses
    weight_t *plastic_words = _plastic_synapses(plastic_region_address);
    const control_t *control_words = synapse_row_plastic_controls(
        fixed_region_address);
    size_t plastic_synapse = synapse_row_num_plastic_controls(
        fixed_region_address);
    const pre_event_history_t *event_history = _plastic_event_history(
        plastic_region_address);

    log_debug("Plastic region %u synapses\n", plastic_synapse);

    // Loop through plastic synapses
    for (uint32_t i = 0; i < plastic_synapse; i++) {

        // Get next weight and control word (auto incrementing control word)
        uint32_t weight = *plastic_words++;
        uint32_t control_word = *control_words++;
        uint32_t synapse_type = synapse_row_sparse_type(control_word);

        log_debug("%08x [%3d: (w: %5u (=", control_word, i, weight);
        synapses_print_weight(
            weight, ring_buffer_to_input_buffer_left_shifts[synapse_type]);
        log_debug("nA) d: %2u, %s, n = %3u)] - {%08x %08x}\n",
                  synapse_row_sparse_delay(control_word),
                  synapse_types_get_type_char(synapse_row_sparse_type(control_word)),
                  synapse_row_sparse_index(control_word), SYNAPSE_DELAY_MASK,
                  SYNAPSE_TYPE_INDEX_BITS);
    }
#endif // LOG_LEVEL >= LOG_DEBUG
}

//---------------------------------------
static inline index_t _sparse_axonal_delay(uint32_t x) {
    return ((x >> SYNAPSE_DELAY_TYPE_INDEX_BITS) & SYNAPSE_AXONAL_DELAY_MASK);
}

bool synapse_dynamics_initialise(
        address_t address, uint32_t n_neurons,
        uint32_t *ring_buffer_to_input_buffer_left_shifts) {

    // Load timing dependence data
    address_t weight_region_address = timing_initialise(address);
    if (address == NULL) {
        return false;
    }

    // Load weight dependence data
    address_t weight_result = weight_initialise(
        weight_region_address, ring_buffer_to_input_buffer_left_shifts);
    if (weight_result == NULL) {
        return false;
    }

    post_event_history = post_events_init_buffers(n_neurons);
    if (post_event_history == NULL) {
        return false;
    }

    return true;
}

bool synapse_dynamics_process_plastic_synapses(
        address_t plastic_region_address, address_t fixed_region_address,
        weight_t *ring_buffers, uint32_t time) {

    // Extract separate arrays of plastic synapses (from plastic region),
    // Control words (from fixed region) and number of plastic synapses
    plastic_synapse_t *plastic_words = _plastic_synapses(
        plastic_region_address);
    const control_t *control_words = synapse_row_plastic_controls(
        fixed_region_address);
    size_t plastic_synapse = synapse_row_num_plastic_controls(
        fixed_region_address);

#ifdef SYNAPSE_BENCHMARK
    num_plastic_pre_synaptic_events += plastic_synapse;
#endif  // SYNAPSE_BENCHMARK

    // Get event history from synaptic row
    pre_event_history_t *event_history = _plastic_event_history(
        plastic_region_address);

    // Get last pre-synaptic event from event history
    const uint32_t last_pre_time = event_history->prev_time;

    // Update pre-synaptic trace
    log_debug("Adding pre-synaptic event to trace at time:%u", time);
    event_history->prev_time = time;

    // Loop through plastic synapses
    for (; plastic_synapse > 0; plastic_synapse--) {

        // Get next control word (auto incrementing)
        uint32_t control_word = *control_words++;

        // Extract control-word components
        // **NOTE** cunningly, control word is just the same as lower
        // 16-bits of 32-bit fixed synapse so same functions can be used
        uint32_t delay_axonal = 0;    //_sparse_axonal_delay(control_word);
        uint32_t delay_dendritic = synapse_row_sparse_delay(control_word);
        uint32_t type = synapse_row_sparse_type(control_word);
        uint32_t index = synapse_row_sparse_index(control_word);
        uint32_t type_index = synapse_row_sparse_type_index(control_word);

        // Create update state from the plastic synaptic word
        update_state_t current_state = synapse_structure_get_update_state(
            *plastic_words, type);

        // Update the synapse state
        final_state_t final_state = _plasticity_update_synapse(
            time, last_pre_time, delay_dendritic, delay_axonal,
            current_state, &post_event_history[index]);

        // Convert into ring buffer offset
        uint32_t ring_buffer_index = synapses_get_ring_buffer_index_combined(
                delay_axonal + delay_dendritic + time, type_index);

        // Add weight to ring-buffer entry
        // **NOTE** Dave suspects that this could be a
        // potential location for overflow
        ring_buffers[ring_buffer_index] += synapse_structure_get_final_weight(
            final_state);

        // Write back updated synaptic word to plastic region
        *plastic_words++ = synapse_structure_get_final_synaptic_word(
            final_state);
    }
    return true;
}

void synapse_dynamics_process_post_synaptic_event(
        uint32_t time, index_t neuron_index) {
    use(time);
    use(neuron_index);
}

void synapse_dynamics_process_target_synaptic_event(
        uint32_t time, index_t neuron_index, uint8_t weight) {

    // learningNow shows currently:
    //                              learning pattern is occurring (=1) or
    //                              learning pattern is NOT occurring (=0) or
    static uint8_t learningNow = 0;

    // the starting time for a target range
    static uint32_t rangeStart = 0;

    // number of times the neuron spiked during correct region
    static int16_t spikeONtarget  = 0; // spike on target from output neuron
    static int16_t spikeOFFtarget = 0; // spike off target from output neuron

    log_debug("Adding post-synaptic event to trace at time:%u", time);

    // Get post-event history
    post_event_history_t *history = &post_event_history[neuron_index];

    // The synaptic signals that can be added to post-events:
    // (weight=1)  syn_signal = 1  # spike from target to output layer
    // (weight=2)  syn_signal = 2  # spike from output neuron back onto itself
    // (weight=3)  syn_signal = 3  # spike from target to previous layer
    // (weight=4)  syn_signal = 4  # spike from output neuron to previous layer
    // (weight=5)  syn_signal = 5  # spike from hidden neuron back onto itself
    // (weight=6)  syn_signal = 6  # starting learning
    // (weight=7)  syn_signal = 7  # ending learning
    // (weight=8)  syn_signal = 8  # ending   target range to output   layer
    // (weight=9)  syn_signal = 9  # ending   target range to previous layer
    // (weight=10) syn_signal = 10 # starting target range to output   layer
    // (weight=11) syn_signal = 11 # starting target range to previous layer

    // if a learning pattern is occurring
    if (learningNow == 1)
    {

        switch(weight) 
        {

            case 2: // spike output neuron back onto itself
            case 4: // spike output neuron to previous layer

                // if range learning is ongoing
                if (rangeStart > 0) 
                {
                    // output spiked during range
                    spikeONtarget += 1;
                }

                else 
                {
                    // output spiked outside of range
                    spikeOFFtarget += 1;
                }


//io_printf(IO_BUF,"%dms, spike from output neuron.    rangeStart:%d    spikeONtarget:%d    spikeOFFtarget:%d\n", time, rangeStart, spikeONtarget, spikeOFFtarget);

                // add post-event
                post_events_add(time, history, weight);

                break;



            // learning pattern ends now
            case 7: 

                // Turn off Learning
                learningNow = 0;
//io_printf(IO_BUF,"%dms, learning pattern ends now. spikeONtarget:%d    spikeOFFtarget:%d\n", time, spikeONtarget, spikeOFFtarget);
                // if there were more on target spikes than off target spikes
                if (spikeONtarget > spikeOFFtarget)
                {
                    // add post-event to stop learning without weight update
                    post_events_add(time, history, 12);
                }

                else 
                {
                    // add post-event to stop learning with weight update
                    post_events_add(time, history, 7);
                }

                break;

            // ending target range to output layer
            case 8: 

                // if the neuron did not spike during target range
                if (spikeONtarget == 0) 
                {
//io_printf(IO_BUF,"\tAt %dms, inside synapse_dynamics...()! learningNow=1,   weight=8,    neuronSpike=0,     with time=%dms\n", time, (time -((time-rangeStart+1)>>1)));

                    // add post-event
                    post_events_add(time, history, 8);// -((time-rangeStart+1)>>1)), history, 8);
                }

                // end the starting target range time
                rangeStart = 0;

                break;

            // ending target range to previous layer
            case 9: 

                // if the neuron spiked during target range
                if (spikeONtarget == 0) 
                {
                    // add post-event
                    post_events_add(time, history, 9); // -((time-rangeStart+1)>>1)), history, 9);
                }

                // end the starting target range time
                rangeStart = 0;

                break;

            // starting target range to either layer
            case 10: 
            case 11:
                // make sure learning is on
                learningNow = 1;

                // record the starting target range time
                rangeStart = time;

                break;

            // otherwise, the weight is 1 - 6
            default:
//if (weight==1)
//io_printf(IO_BUF,"%dms, spike from target to output neuron.    rangeStart:%d    spikeONtarget:%d    spikeOFFtarget:%d\n", time, rangeStart, spikeONtarget, spikeOFFtarget);

                // add post-event
                post_events_add(time, history, weight);
        }
    }

    // as of last ms, a learning pattern is not occuring
    else
    {
        // learning pattern starts now
        if (weight == 6)
        {
            // Turn on Learning
            learningNow = 1;

            // reset output neuron spiked
            spikeONtarget  = 0; // spike on target from output neuron
            spikeOFFtarget = 0; // spike off target from output neuron

            // add post-event
            post_events_add(time, history, weight);
        }
    }
    //io_printf(IO_BUF,"\tAt %dms, in synapse_dynamics_process_target_synaptic_event(), learningNow=%d\tweight=%d\n", time, learningNow, weight);
}


input_t synapse_dynamics_get_intrinsic_bias(uint32_t time, index_t neuron_index) {
    use(time);
    use(neuron_index);
    return 0.0k;
}

uint32_t synapse_dynamics_get_plastic_pre_synaptic_events(){
#ifdef SYNAPSE_BENCHMARK
    return num_plastic_pre_synaptic_events;
#else
    return 0;
#endif  // SYNAPSE_BENCHMARK
}
