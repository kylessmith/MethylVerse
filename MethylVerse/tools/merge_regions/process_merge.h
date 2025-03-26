//=====================================================================================
// Common structs, parameters, functions
// by Kyle S. Smith
//-------------------------------------------------------------------------------------
#ifndef __MERGE_REGIONS_H__
#define __MERGE_REGIONS_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/ailist/augmented_interval_list.h"
#include "src/labeled_aiarray/labeled_augmented_array.h"


ailist_t *ailist_adjacent_merge(ailist_t *ail, uint32_t n);

labeled_aiarray_t *adjacent_merge(labeled_aiarray_t *laia, uint32_t n);

labeled_aiarray_t *adjacent_merge_less_than(labeled_aiarray_t *laia, double *values, double cutoff);

labeled_aiarray_t *adjacent_merge_greater_than(labeled_aiarray_t *laia, double *values, double cutoff);

#endif