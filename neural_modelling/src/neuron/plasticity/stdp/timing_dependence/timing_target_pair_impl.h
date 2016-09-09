/*
 * The supervised learning rule is implemented in this file.
 * s = Tar-Pre or Out-Pre at times of Tar and Out
 * PostSynaptic Potential (PSP) at s = exp**(-s/tauMem)
 * where tauMem and tauSyn are time constants of neuron and synapse.
 * At learning pattern end,
 * all the PSP from Tar are summed and subtracted from all summed Out PSP.
 *
 * To implement the rule in this file...
 * we have timing_apply_post_spike(...) parameters
 * time = Tar or Out
 * last_pre_time = Pre
 * post_trace_t = action potential, denotes if Tar (=0) or Out (=1)
 * time_since_last_pre = s
 * DECAY_LOOKUP_TAU_PLUS( time_since_last_pre) = exp**(-s/tauMem)
 * previous_state.accumulator = accumulated prior updates
 * previous_state.weight_state = weight change at end of learning pattern
 *
 */

#ifndef _TIMING_TARGET_PAIR_IMPL_H_
#define _TIMING_TARGET_PAIR_IMPL_H_

//---------------------------------------
// Includes
//---------------------------------------
#include "../synapse_structure/synapse_structure_weight_target.h"
#include "../synapse_structure/synapse_structure.h"
#include "../weight_dependence/weight_one_term.h"
//#include "timing.h"
//#include "../../common/maths.h"
//#include "../../common/stdp_typedefs.h"
//#include "../../../synapse_row.h"

// Include debug header for log_info etc
#include <debug.h>

// Include generic plasticity maths functions
#include "../../common/maths.h"
#include "../../common/stdp_typedefs.h"

//---------------------------------------
// Macros
//---------------------------------------
// Exponential decay lookup parameters
#define TAU_PLUS_TIME_SHIFT 0
#define TAU_PLUS_SIZE 256

// Helper macros for looking up decays
#define DECAY_LOOKUP_TAU_PLUS(time) \
    maths_lut_exponential_decay( \
        time, TAU_PLUS_TIME_SHIFT, TAU_PLUS_SIZE, tau_plus_lookup)

//---------------------------------------
// Externals
//---------------------------------------
extern int16_t tau_plus_lookup[TAU_PLUS_SIZE];

//---------------------------------------
// Typedefines
//---------------------------------------
typedef int16_t post_trace_t;
static post_trace_t timing_get_initial_post_trace();
address_t timing_initialise(address_t address);
static update_state_t pattern_begins(update_state_t previous_state);
static update_state_t timing_apply_post_spike(uint32_t time_since_last_pre,
        post_trace_t syn_signal, update_state_t previous_state);
static update_state_t pattern_ends(update_state_t previous_state);

//---------------------------------------
// Timing dependence inline functions
//---------------------------------------
static inline post_trace_t timing_get_initial_post_trace() {
    return 0;
}

    // The synaptic signals that can be added to post-events:
    // syn_signal = 1   # spike from target to output layer
    // syn_signal = 2   # spike from output neuron back onto itself
    // syn_signal = 3   # spike from target to previous layer
    // syn_signal = 4   # spike from output neuron to previous layer
    // syn_signal = 5   # spike from hidden neuron back onto itself
    // syn_signal = 6   # starting learning
    // syn_signal = 7   # ending learning

//---------------------------------------
// This is called when a learning pattern begins as a 
// result of a triplet passing through a Target synapse
static update_state_t pattern_begins(update_state_t previous_state)
{
    previous_state.accumulator = 0; // reset accumulator to baseline

    return previous_state;
}

//---------------------------------------
// This will apply an actual postsynaptic spike
// usefull variables:
// time          = postsynaptic (+ dendritic delay) or target spike time
// last_pre_time = last presynaptic spike time
static inline update_state_t timing_apply_post_spike(
        uint32_t time_since_last_pre, post_trace_t syn_signal,
        update_state_t previous_state)
{
    switch(syn_signal)
    {

        
        case 1: // Learning is on and a spike from target to output layer
        case 3: // Learning is on and a spike from target to hidden layer
//io_printf(IO_BUF,"Increasing accumulator from %d    ", previous_state.accumulator);
            // record and add future synaptic update to accumulator
            previous_state.accumulator += DECAY_LOOKUP_TAU_PLUS(time_since_last_pre);
//io_printf(IO_BUF,"to %d    using time:%dms\n", previous_state.accumulator, time_since_last_pre);
            break;

        case 2: // Learning is on and a spike from output neuron back onto itself
        case 4: // Learning is on and a spike from output neuron to hidden layer
//io_printf(IO_BUF,"Decreasing accumulator from %d    ", previous_state.accumulator);
            // update the accumulation
            previous_state.accumulator -= DECAY_LOOKUP_TAU_PLUS(time_since_last_pre);
//io_printf(IO_BUF,"to %d    using time:%dms\n", previous_state.accumulator, time_since_last_pre);
    } // end switch

    return previous_state;
}

//---------------------------------------
// This is called when a learning pattern ends as a result of
// a doublet passing through a Target synapse during learning
static update_state_t pattern_ends(update_state_t previous_state)
{
//if (previous_state.accumulator != 0)
//    io_printf(IO_BUF,"Updating weight state with accumulator = %d    ", previous_state.accumulator);

    // Apply potentiation to state (which is a weight_state) if positive
    if (previous_state.accumulator > 0)
    {
//io_printf(IO_BUF," (initial_weight before:%d,", previous_state.weight_state.initial_weight);
        // update weight
        previous_state.weight_state.initial_weight = 
                         weight_apply_potentiation( previous_state.weight_state,
                                                    previous_state.accumulator);
//io_printf(IO_BUF," to:%d\n", previous_state.weight_state.initial_weight);
    }

    // Apply depression to state (which is a weight_state) if negative
    else if (previous_state.accumulator < 0)
    {
//io_printf(IO_BUF," (initial_weight before:%d,", previous_state.weight_state.initial_weight);
        // update weight
        previous_state.weight_state.initial_weight = 
                           weight_apply_depression( previous_state.weight_state,
                                                    previous_state.accumulator);
//io_printf(IO_BUF," to:%d\n", previous_state.weight_state.initial_weight);
    }

    // reset accumulator to baseline
    previous_state.accumulator = 0; 

    return previous_state;
}

#endif // _TIMING_TARGET_PAIR_IMPL_H_
