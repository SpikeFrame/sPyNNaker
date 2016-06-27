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
 * previous_state.accumLast   = current update
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
static update_state_t timing_apply_post_spike(uint32_t time_since_last_pre,
		post_trace_t syn_signal, update_state_t previous_state);

//---------------------------------------
// Timing dependence inline functions
//---------------------------------------
static inline post_trace_t timing_get_initial_post_trace() {
    return 0;
}

// The synaptic signals (syn_signal) that can be sent to post-events:
//   0: Learning is ongoing, accumulate values
// 1-3: A spike passed through a target synapse and
//      1: Turning on Learning
//      2: Learning is ongoing, accumulate values
//      3: The current pattern has come to an end, update weight
//   4: Learning is not happening, do not accumulate weight values

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

	// a spike came from a non-Target neuron
	case 0:

		// io_printf(IO_BUF,"time_since_last_pre: %dms\n", time_since_last_pre);
		log_debug("\t\t\ttime_since_last_pre_event=%u\n",time_since_last_pre);

		// add last synaptic update to accumulation
		previous_state.accumulator += previous_state.accumLast;

		// update the last accumulation
		previous_state.accumLast = -1 * DECAY_LOOKUP_TAU_PLUS( time_since_last_pre);

		break;

	// a doublet passed through a Target synapse, starting learning
	case 1:

		previous_state.accumulator = 0; // reset accumulator to baseline
		previous_state.accumLast   = 0; // reset accumLast to baseline

		break;

	// Learning is on and a spike passed through a Target synapse
	case 2:

		// add last synaptic update to accumulation
		previous_state.accumulator += previous_state.accumLast;

		// update the last accumulation
		previous_state.accumLast = DECAY_LOOKUP_TAU_PLUS( time_since_last_pre);

		break;

	// Learning is on and a doublet passed through a Target synapse
	case 3:

		// Apply potentiation to state (which is a weight_state) if positive
		if (previous_state.accumulator > 0)
		{
			previous_state.weight_state =
					weight_one_term_apply_potentiation(
							previous_state.weight_state,
							previous_state.accumulator);
		}

		// Apply depression to state (which is a weight_state) if negative
		else if (previous_state.accumulator < 0)
		{
			previous_state.weight_state =
					weight_one_term_apply_depression(
							previous_state.weight_state,
							(-1 * previous_state.accumulator));
		}

	} // end switch

	return previous_state;
}


#endif // _TIMING_TARGET_PAIR_IMPL_H_
