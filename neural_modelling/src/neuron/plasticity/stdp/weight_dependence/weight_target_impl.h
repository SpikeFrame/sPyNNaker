#ifndef _WEIGHT_TARGET_IMPL_H_
#define _WEIGHT_TARGET_IMPL_H_

// Include generic plasticity maths functions
#include "../../common/maths.h"
#include "../../common/stdp_typedefs.h"
#include "../../../synapse_row.h"

#include <debug.h>

//---------------------------------------
// Structures
//---------------------------------------
typedef struct {
    int32_t min_weight;
    int32_t max_weight;

    int32_t a2_plus;
    int32_t a2_minus;
} plasticity_weight_region_data_t;

typedef struct {
    int32_t initial_weight;

    int32_t a2_plus;
    int32_t a2_minus;

    const plasticity_weight_region_data_t *weight_region;
} weight_state_t;

//#include "weight_one_term.h"

//---------------------------------------
// Declarations
//---------------------------------------
static inline int32_t weight_apply_depression(   weight_state_t state, 
                                                 int32_t decrease );
static inline int32_t weight_apply_potentiation( weight_state_t state,
                                                 int32_t increase );

//---------------------------------------
// Externals
//---------------------------------------
extern plasticity_weight_region_data_t
    plasticity_weight_region_data[SYNAPSE_TYPE_COUNT];

//---------------------------------------
// STDP weight dependance functions
//---------------------------------------
static inline weight_state_t weight_get_initial(weight_t weight,
                                                index_t synapse_type) {

    return (weight_state_t ) {
        .initial_weight = (int32_t) weight,
        .a2_plus = 0,
        .a2_minus = 0,
        .weight_region = &plasticity_weight_region_data[synapse_type]
    };
}

//---------------------------------------
static inline int32_t weight_apply_depression( weight_state_t state,  
                                               int32_t decrease) 
{

        //io_printf(IO_BUF,"Decreasing synaptic strength from %d", state.initial_weight);

        // pre-scale accumulator into weight format
        //int32_t new_weight = STDP_FIXED_MUL_16X16( decrease, 
        //                                state.weight_region->a2_minus);


        // Multiply lower 16-bits together then shift down
        int32_t new_weight = (__smulbb(decrease, state.weight_region->a2_minus)) >> 11;

    

        //io_printf(IO_BUF," (new_weightA:%d    ",  new_weight);

        // Apply depression (negative new_weight) to initial weight
        new_weight += state.initial_weight;

        //io_printf(IO_BUF," (decrease:%d * a2_minus:%d)    new_weightB:%d    min:%d    max:%d    clamped=%d)    ", decrease, state.weight_region->a2_minus, new_weight, state.weight_region->min_weight, state.weight_region->max_weight, MIN(state.weight_region->max_weight,MAX(new_weight, state.weight_region->min_weight)));

        // Clamp and return new weight
        return MIN(state.weight_region->max_weight,
                   MAX(new_weight, state.weight_region->min_weight));
}

//---------------------------------------
static inline int32_t weight_apply_potentiation( weight_state_t state,
                                                 int32_t increase ) 
{

        //io_printf(IO_BUF,"Increasing synaptic strength from %d", state.weight_region->a2_plus);

        // pre-scale accumulator into weight format
        //int32_t new_weight = STDP_FIXED_MUL_16X16( increase, 
        //                               state.weight_region->a2_plus);

        // Multiply lower 16-bits together then shift down
        int32_t new_weight = (__smulbb(increase, state.weight_region->a2_plus)) >> 11;

//io_printf(IO_BUF," (new_weightA:%d)", new_weight); 


        // Apply potentiation (positive new_weight) to initial weight
        new_weight += state.initial_weight;

//io_printf(IO_BUF," ((increase:%d * a2_plus:%d)    new_weight:%d    min:%d    max:%d    clamped=%d)    ", increase, state.weight_region->a2_plus, new_weight, state.weight_region->min_weight, state.weight_region->max_weight, MIN(state.weight_region->max_weight,MAX(new_weight, state.weight_region->min_weight)));


        // Clamp and return new weight
        return MIN(state.weight_region->max_weight,
                   MAX(new_weight, state.weight_region->min_weight));
}

//---------------------------------------
static inline weight_t weight_get_final(weight_state_t new_state) {

    // Clamp new weight
    int32_t new_weight = MIN(new_state.weight_region->max_weight,
            MAX(new_state.initial_weight, new_state.weight_region->min_weight));

    return (weight_t) new_weight;
}

#endif // _WEIGHT_TARGET_IMPL_H_
