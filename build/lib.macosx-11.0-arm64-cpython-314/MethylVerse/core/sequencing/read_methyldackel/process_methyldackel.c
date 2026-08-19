#include "process_methyldackel.h"


sparse_record_t *sparse_record_init(void)
{   /* Initialize sparse record */
    
    // Initialize variables
    sparse_record_t *sp = (sparse_record_t *)malloc(sizeof(sparse_record_t));
    if (sp == NULL)
    {
        printf("Memory allocation failed");
        exit(1); // exit the program
    }
    sp->size = 0;
    sp->max_size = 64;
    sp->indices = (long *)malloc(sizeof(long) * 64);
    if (sp->indices == NULL)
    {
        printf("Memory allocation failed");
        exit(1); // exit the program
    }

    sp->values = (double *)malloc(sizeof(double) * 64);
    if (sp->values == NULL)
    {
        printf("Memory allocation failed");
        exit(1); // exit the program
    }

    return sp;
}


void sparse_record_destroy(sparse_record_t *sp)
{   /* Free sparse_record memory */

    free(sp->indices);
    free(sp->values);
    free(sp);

    return;
}


void sparse_record_add(sparse_record_t *sp, uint32_t index, double value)
{   /* Add interval to sparse record */

    // Check if size needs to be increased
    push_sparse(sp, index, value);

    return;
}


char *parse_bed(char *s, int32_t *st_, int32_t *en_)
{
	char *p, *q, *ctg = 0;
	int32_t i, st = -1, en = -1;

	for (i = 0, p = q = s;; ++q)
    {
		if (*q == '\t' || *q == '\0')
        {
			int c = *q;
			*q = 0;
			if (i == 0)
            {
                ctg = p;
            }
			else if (i == 1)
            {
                st = atol(p);
            }
			else if (i == 2)
            {
                en = atol(p);
            }

			++i;
            p = q + 1;
			if (c == '\0')
            {
                break;
            }
		}
	}

	*st_ = st;
    *en_ = en;

	return i >= 3? ctg : 0;
}


char *parse_methyldackel(char *s, int32_t *st_, int32_t *en_, int32_t *m_, int32_t *u_)
{
	char *p, *q, *ctg = 0;
	int32_t i, st = -1, en = -1, u = -1, m = -1;

	for (i = 0, p = q = s;; ++q)
    {
		if (*q == '\t' || *q == '\0')
        {
			int c = *q;
			*q = 0;
			if (i == 0)
            {
                ctg = p;
            }
			else if (i == 1)
            {
                st = atol(p);
            }
			else if (i == 2)
            {
                en = atol(p);
            }
            else if (i == 4)
            {
                m = atol(p);
            }
            else if (i == 5)
            {
                u = atol(p);
            }

			++i;
            p = q + 1;
			if (c == '\0')
            {
                break;
            }
		}
	}

	*st_ = st;
    *en_ = en;
    *u_ = u;
    *m_ = m;

	return i >= 3? ctg : 0;
}


labeled_aiarray_t *read_bed(const char* fn)
{   //faster than strtok()
	gzFile fp;
	labeled_aiarray_t *laia;
	kstream_t *ks;
	kstring_t str = {0,0,0};
	int32_t k = 0;

	if ((fp = gzopen(fn, "r")) == 0)
    {
		return 0;
    }

	ks = ks_init(fp);
	laia = labeled_aiarray_init();
    
	while (ks_getuntil(ks, KS_SEP_LINE, &str, 0) >= 0)
    {
		char *name;
		int32_t st, en;
		name = parse_bed(str.s, &st, &en);
		if (name)
        {
            labeled_aiarray_add(laia, st, en, name);
            k++;
        }
	}

	free(str.s);
	ks_destroy(ks);
	gzclose(fp);

	return laia;
}


labeled_aiarray_t *find_all_intervals(const char file_names[], int length, int str_len)
{
    // Iterate over itervals and add
    labeled_aiarray_t *total_laia;
    int i;
    int total_length = length * str_len;
    for (i = 0; i < total_length; i+=str_len)
    {
        char file_name[str_len + 1];
        slice_str(file_names, file_name, i, i+str_len);
        labeled_aiarray_t *laia = read_bed(file_name);

        if (i == 0)
        {
            total_laia = labeled_aiarray_copy(laia);
        }
        else
        {
            uint8_t *has_match = (uint8_t *)malloc(sizeof(uint8_t) * laia->total_nr);
            // Check if memory was allocated
            if (has_match == NULL)
            {
                fprintf (stderr, "Out of memory!!! (has_match)\n");
                exit(1);
            }
            labeled_aiarray_has_exact_match(laia, total_laia, &has_match[0]);

            // Iterate over matches and add
            labeled_aiarray_iter_t *iter = labeled_aiarray_iter_init(laia);
            while (labeled_aiarray_iter_next(iter) != 0)
            {
                if (has_match[iter->n] == 0)
                {
                    labeled_aiarray_add(total_laia, iter->intv->i->start, iter->intv->i->end, iter->intv->name);
                }
            }

            // Free
            labeled_aiarray_iter_destroy(iter);
            free(has_match);        }
    }
    
    return total_laia;
}


void insert_betas(const char* fn, double *betas, labeled_aiarray_t *laia)
{   //faster than strtok()
    
    gzFile fp;
	kstream_t *ks;
	kstring_t str = {0,0,0};
	int32_t k = 0;
    int index;

	if ((fp = gzopen(fn, "r")) == 0)
    {
		return;
    }

	ks = ks_init(fp);
    
	while (ks_getuntil(ks, KS_SEP_LINE, &str, 0) >= 0)
    {
		char *name;
		int32_t st, en, u, m;
		name = parse_methyldackel(str.s, &st, &en, &m, &u);
		if (name)
        {
            index = labeled_aiarray_where_interval(laia, name, st, en);
            if (index != -1)
            {
                if (m == 0)
                {
                    betas[index] = 0.0;
                }
                else if (u == 0)
                {
                    betas[index] = 1.0;
                }
                else
                {
                    betas[index] = (double)m / ((double)m + (double)u);
                }
            }
            k++;
        }
	}

	free(str.s);
	ks_destroy(ks);
	gzclose(fp);

	return;
}


sparse_record_t *insert_sparse_betas(const char* fn, labeled_aiarray_t *laia)
{   //faster than strtok()
    
    gzFile fp;
	kstream_t *ks;
	kstring_t str = {0,0,0};
	int32_t k = 0;
    int index;
    double beta;
    sparse_record_t *sparse_betas = sparse_record_init();

	if ((fp = gzopen(fn, "r")) == 0)
    {
		return sparse_betas;
    }

	ks = ks_init(fp);
    
	while (ks_getuntil(ks, KS_SEP_LINE, &str, 0) >= 0)
    {
		char *name;
		int32_t st, en, u, m;
		name = parse_methyldackel(str.s, &st, &en, &m, &u);
		if (name)
        {
            index = labeled_aiarray_where_interval(laia, name, st, en);
            if (index != -1)
            {
                if (m == 0)
                {
                    beta = 0.0;
                    //betas[index] = 0.0;
                }
                else if (u == 0)
                {
                    beta = 1.0;
                    //betas[index] = 1.0;
                }
                else
                {
                    beta = (double)m / ((double)m + (double)u);
                    //betas[index] = (double)m / ((double)m + (double)u);
                }

                sparse_record_add(sparse_betas, index, beta);
            }
            k++;
        }
	}

	free(str.s);
	ks_destroy(ks);
	gzclose(fp);

	return sparse_betas;
}


sparse_record_t *insert_sparse_coverage(const char* fn, labeled_aiarray_t *laia)
{   //faster than strtok()
    
    gzFile fp;
	kstream_t *ks;
	kstring_t str = {0,0,0};
	int32_t k = 0;
    int index;
    double cov;
    sparse_record_t *sparse_covs = sparse_record_init();

	if ((fp = gzopen(fn, "r")) == 0)
    {
		return sparse_covs;
    }

	ks = ks_init(fp);
    
	while (ks_getuntil(ks, KS_SEP_LINE, &str, 0) >= 0)
    {
		char *name;
		int32_t st, en, u, m;
		name = parse_methyldackel(str.s, &st, &en, &m, &u);
		if (name)
        {
            index = labeled_aiarray_where_interval(laia, name, st, en);
            if (index != -1)
            {
                cov = (double)m + (double)u;

                sparse_record_add(sparse_covs, index, cov);
            }
            k++;
        }
	}

	free(str.s);
	ks_destroy(ks);
	gzclose(fp);

	return sparse_covs;
}