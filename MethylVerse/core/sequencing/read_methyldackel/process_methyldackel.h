//=====================================================================================
// Common structs, parameters, functions
// Original: https://github.com/databio/aiarray/tree/master/src
// by Kyle S. Smith
//-------------------------------------------------------------------------------------
#ifndef __PROCESS_METHYLDACKEL_H__
#define __PROCESS_METHYLDACKEL_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <zlib.h>
#include "src/labeled_aiarray/labeled_augmented_array.h"
#include "kseq.h"


typedef struct {
	int size;							// Current size
    int max_size;						// Maximum size
	double *values;			            // Store values
	long *indices;						// Store indices
} sparse_record_t;


sparse_record_t *sparse_record_init(void);

void sparse_record_destroy(sparse_record_t *sp);

void sparse_record_add(sparse_record_t *sp, uint32_t index, double value);

char *parse_bed(char *s, int32_t *st_, int32_t *en_);

labeled_aiarray_t *read_bed(const char* fn);

labeled_aiarray_t *find_all_intervals(const char file_names[], int length, int str_len);

char *parse_methyldackel(char *s, int32_t *st_, int32_t *en_, int32_t *m_, int32_t *u_);

void insert_betas(const char* fn, double *betas, labeled_aiarray_t *laia);

sparse_record_t *insert_sparse_betas(const char* fn, labeled_aiarray_t *laia);

sparse_record_t *insert_sparse_coverage(const char* fn, labeled_aiarray_t *laia);

KSTREAM_INIT(gzFile, gzread, 0x10000)


#define push_sparse(sp, index, value) do {											\
    if (sp->size == sp->max_size){													        \
		sp->max_size = sp->max_size? sp->max_size<<1 : 2;							        \
		sp->indices = (long*)realloc(sp->indices, sizeof(long) * sp->max_size);		\
        sp->values = (double*)realloc(sp->values, sizeof(double) * sp->max_size);	\
	}																				        \
	sp->indices[sp->size] = (index);												    \
    sp->values[sp->size++] = (value);												    \
} while (0)


#endif