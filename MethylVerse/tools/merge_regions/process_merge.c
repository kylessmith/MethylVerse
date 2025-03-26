#include "process_merge.h"


ailist_t *ailist_adjacent_merge(ailist_t *ail, uint32_t n)
{   /* Merge intervals in constructed ailist_t object */

    // Initialize merged atrributes
    int previous_end = ail->interval_list[0].end;
    int previous_start = ail->interval_list[0].start;
    int previous_id = ail->interval_list[0].id_value;
    ailist_t *merged_list = ailist_init();
    interval_t *intv;
    uint32_t n_count = 0;

    // Create sorted iterators
    ailist_sorted_iter_t* ail_iter = ailist_sorted_iter_init(ail);
    intv = ail_iter->intv;
    while (ailist_sorted_iter_next(ail_iter) != 0)
    {
        intv = ail_iter->intv;

        if (n_count < n)
        {
            previous_end = MAX(previous_end, (int)intv->end);
            n_count++;
        }
        else
        {
            ailist_add(merged_list, previous_start, previous_end, previous_id);
            previous_start = intv->start;
            previous_end = intv->end;
            previous_id = intv->id_value;
            n_count = 0;
        }
    }

    // Destroy iterator
    ailist_sorted_iter_destroy(ail_iter);

    // Add last interval
    ailist_add(merged_list, previous_start, previous_end, previous_id);

    return merged_list;
}


labeled_aiarray_t *adjacent_merge(labeled_aiarray_t *laia, uint32_t n)
{   /* Merge nearby intervals */

    // Initialize label_aiarray
    labeled_aiarray_t *merged_laia = labeled_aiarray_init();

    // Iterate over labels
    int32_t i;
    for (i = 0; i < laia->n_labels; i++)
    {
        label_t *p = &laia->labels[i];
        ailist_t *merged_ail = ailist_adjacent_merge(p->ail, n);
        labeled_aiarray_wrap_ail(merged_laia, merged_ail, p->name);
    }

    // Re-sort
    labeled_aiarray_order_sort(merged_laia);

    return merged_laia;
}


labeled_aiarray_t *adjacent_merge_less_than(labeled_aiarray_t *laia, double *values, double cutoff)
{   

    char *previous_label = NULL;
    int previous_start = 0;
    int previous_end = 0;
    int merging = 0;

    labeled_aiarray_t *merged_laia = labeled_aiarray_init();
    labeled_aiarray_iter_t *laia_iter = labeled_aiarray_iter_init(laia);
    int i = 0;
    while(labeled_aiarray_iter_next(laia_iter) != 0)
    {
        double value = values[i];

        if (previous_label && strcmp(previous_label, laia_iter->intv->name) == 0)
        {
            if (value < cutoff)
            {
                if (merging == 1)
                {
                    previous_start = MIN(previous_start, laia_iter->intv->i->start);
                    previous_end = MAX(previous_end, laia_iter->intv->i->end);
                }
                else {
                    previous_start = laia_iter->intv->i->start;
                    previous_end = laia_iter->intv->i->end;
                }
                
                merging = 1;
            }
            else {
                if (merging == 1)
                {
                    labeled_aiarray_add(merged_laia, previous_start, previous_end, previous_label);
                    previous_start = laia_iter->intv->i->start;
                    previous_end = laia_iter->intv->i->end;
                    merging = 0;
                }
            }
        }
        else {
            if (merging == 1)
            {
                labeled_aiarray_add(merged_laia, previous_start, previous_end, previous_label);
            }
            previous_label = laia_iter->intv->name;
            previous_start = laia_iter->intv->i->start;
            previous_end = laia_iter->intv->i->end;

            if (value < cutoff)
            {
                merging = 1;
            }
            else {
                merging = 0;
            }
        }

        i++;
    }

    if (merging == 1)
    {
        labeled_aiarray_add(merged_laia, previous_start, previous_end, previous_label);
    }

    // Destroy iterator
    labeled_aiarray_iter_destroy(laia_iter);

    return merged_laia;
}


labeled_aiarray_t *adjacent_merge_greater_than(labeled_aiarray_t *laia, double *values, double cutoff)
{   

    char *previous_label = NULL;
    int previous_start = 0;
    int previous_end = 0;
    int merging = 0;

    labeled_aiarray_t *merged_laia = labeled_aiarray_init();
    labeled_aiarray_iter_t *laia_iter = labeled_aiarray_iter_init(laia);
    int i = 0;
    while(labeled_aiarray_iter_next(laia_iter) != 0)
    {
        double value = values[i];

        if (previous_label && strcmp(previous_label, laia_iter->intv->name) == 0)
        {
            if (value > cutoff)
            {
                if (merging == 1)
                {
                    previous_start = MIN(previous_start, laia_iter->intv->i->start);
                    previous_end = MAX(previous_end, laia_iter->intv->i->end);
                }
                else {
                    previous_start = laia_iter->intv->i->start;
                    previous_end = laia_iter->intv->i->end;
                }
                
                merging = 1;
            }
            else {
                if (merging == 1)
                {
                    labeled_aiarray_add(merged_laia, previous_start, previous_end, previous_label);
                    previous_start = laia_iter->intv->i->start;
                    previous_end = laia_iter->intv->i->end;
                    merging = 0;
                }
            }
        }
        else {
            if (merging == 1)
            {
                labeled_aiarray_add(merged_laia, previous_start, previous_end, previous_label);
            }
            previous_label = laia_iter->intv->name;
            previous_start = laia_iter->intv->i->start;
            previous_end = laia_iter->intv->i->end;

            if (value > cutoff)
            {
                merging = 1;
            }
            else {
                merging = 0;
            }
        }

        i++;
    }

    if (merging == 1)
    {
        labeled_aiarray_add(merged_laia, previous_start, previous_end, previous_label);
    }

    // Destroy iterator
    labeled_aiarray_iter_destroy(laia_iter);

    return merged_laia;
}