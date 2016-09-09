// The supervised (target) algorithm requires weight updates to be accumulated
// and only applied (modify synapse) at the end of a target pattern. Therefore, 
// this class copies and adds an 'accumulator' variable to the original 
// synapse_structure_weight.h file for this purpose. -Eric

#ifndef _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_
#define _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_

//---------------------------------------
// Structures
//---------------------------------------

// Plastic synapse types are just weights;
typedef struct plastic_synapse_t {
    weight_t weight;
    int16_t accumulator; // accumulates updates for later synapse modification
} plastic_synapse_t;

// The update state is a weight state with 32-bit ARM-friendly versions of the
// accumulator and the state
typedef struct update_state_t {
    weight_state_t weight_state;
    int32_t accumulator; // accumulate updates for later synapse modification
} update_state_t;

// The final state is just a weight as this is
// both the weight and the synaptic word
typedef plastic_synapse_t final_state_t;

//---------------------------------------
// Synapse interface functions
//---------------------------------------
static inline update_state_t synapse_structure_get_update_state(
        plastic_synapse_t synaptic_word, index_t synapse_type) {

    // Create update state, using weight dependence to initialise the weight
    // state and copying other parameters from the synaptic word into 32-bit
    // form
    update_state_t update_state;
    update_state.weight_state = weight_get_initial(synaptic_word.weight,
                                                   synapse_type);
    update_state.accumulator = (int32_t) synaptic_word.accumulator;

    return update_state;
}

//---------------------------------------
static inline final_state_t synapse_structure_get_final_state(
        update_state_t state) {

    // Get weight from state
    //weight_t weight = weight_get_final(state.weight_state);

    // Build this into synaptic word along with updated accumulator and state
    return (final_state_t) {
        .weight = (weight_t) state.weight_state.initial_weight,
        .accumulator = (int16_t) state.accumulator,
    };
}

//---------------------------------------
static inline weight_t synapse_structure_get_final_weight(
        final_state_t final_state) {
    return final_state.weight;
}

//---------------------------------------
static inline plastic_synapse_t synapse_structure_get_final_synaptic_word(
        final_state_t final_state) {
    return final_state;
}

#endif  // _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_
