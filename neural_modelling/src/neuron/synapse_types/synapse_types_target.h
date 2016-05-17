/*
 * Brief implementation of synapse_types.h for Target shaping
*
* This is a 'supervisor' synapse that provides a target spike time
* and does NOT provide any postsynaptic neuron input.
*/


#ifndef _SYNAPSE_TYPES_TARGET_
#define _SYNAPSE_TYPES_TARGET_

//---------------------------------------
// Macros
//---------------------------------------
#define SYNAPSE_TYPE_BITS 2
#define SYNAPSE_TYPE_COUNT 3

#include <debug.h>

//---------------------------------------
// Synapse parameters
//---------------------------------------
#include "synapse_types.h"

typedef enum input_buffer_regions TARGET;

//---------------------------------------
// Synapse shaping inline implementation
//---------------------------------------


//! \brief returns a human readable character for the type of synapse.
//! T = target type of synapse.
//! \param[in] synapse_type_index the synapse type index
//! (there is a specific index interpretation in each synapse type)
//! \return a human readable character representing the synapse type.
static inline const char *synapse_types_get_type_char(
        index_t synapse_type_index) {
    if (synapse_type_index == TARGET)  {
        return "T";
    } else {
        log_debug("did not recognise synapse type %i", synapse_type_index);
        return "?";
    }
}


#endif  // _SYNAPSE_TYPES_TARGET_
