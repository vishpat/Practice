#include <stdlib.h>
#include "vp_list.h"
#include "vp_queue.h"

vp_queue_t*
vp_queue_create()
{
    return vp_list_create();
}

void 
vp_queue_free(vp_queue_t *queue)
{
    vp_list_free(queue, NULL);
}

void
vp_queue_enqueue(vp_queue_t *q, void *item)
{
    vp_list_add_tail(q, item);
}

void*
vp_queue_dequeue(vp_queue_t *q)
{
    return vp_list_remove_head(q);
}

bool
vp_queue_empty(vp_queue_t *q)
{
    return vp_list_empty(q);
}
