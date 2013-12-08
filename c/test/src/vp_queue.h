#ifndef _VP_QUEUE_H_
#define _VP_QUEUE_H_
#include <stdbool.h>

typedef struct vp_list_s vp_queue_t;

vp_queue_t *vp_queue_create();
void vp_queue_enqueue(vp_queue_t *, void *);
void *vp_queue_dequeue(vp_queue_t *);
bool vp_queue_empty(vp_queue_t *);

#endif

