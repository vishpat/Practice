#ifndef _VP_LIST_H_
#define _VP_LIST_H_

#include <stdbool.h>

typedef struct vp_list_s vp_list_t;
typedef void (*vp_list_node_apply)(void *);

vp_list_t* vp_list_create(void);

bool vp_list_empty(vp_list_t *);

void *vp_list_head(vp_list_t *vp_list);
void *vp_list_tail(vp_list_t *vp_list);

bool vp_list_add_head(vp_list_t *vp_list, void *item);
bool vp_list_add_tail(vp_list_t *vp_list, void *item);

void* vp_list_remove_head(vp_list_t *vp_list);
void* vp_list_remove_tail(vp_list_t *vp_list);

void* vp_list_remove(vp_list_t *vp_list, void *item);
void* vp_list_find(vp_list_t *vp_list, void *item);

void vp_list_apply(vp_list_t *vp_list, vp_list_node_apply apply); 

#endif
