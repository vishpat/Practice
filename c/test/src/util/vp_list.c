#include "vp_list.h"

#include <stdlib.h>
#include <stdio.h>

typedef struct vp_list_node {
    struct vp_list_node *next;
    void *item;
} vp_list_node_t;

struct vp_list_s {
    vp_list_node_t *head;
    vp_list_node_t *tail;
};

vp_list_t *
vp_list_create()
{
    vp_list_t *vp_list_p = (vp_list_t *)malloc(sizeof(*vp_list_p));
    vp_list_p->head = NULL;
    vp_list_p->tail = NULL;
    return vp_list_p;
}

static void
_vp_list_free_all(vp_list_node_t *node, vp_list_node_apply item_free_func)
{
    if (node->next) {
        _vp_list_free_all(node->next, item_free_func);
        node->next = NULL;
    } else {
        if (item_free_func) {
            item_free_func(node->item);
        }
        free(node); 
    }
}

void 
vp_list_free(vp_list_t *vp_list, 
                  vp_list_node_apply item_free_func)
{
    _vp_list_free_all(vp_list->head, item_free_func);
    free(vp_list);
}

bool
vp_list_empty(vp_list_t *vp_list)
{
    return vp_list->head == NULL && vp_list->tail == NULL;
}

void*
vp_list_head(vp_list_t *vp_list)
{
    return vp_list->head ? vp_list->head->item : NULL;
}

void*
vp_list_tail(vp_list_t *vp_list)
{
    return vp_list->tail ? vp_list->tail->item : NULL;
}

static vp_list_node_t*
vp_list_predecessor(vp_list_node_t *vp_list_node_p, void *item)
{
    if (vp_list_node_p == NULL || vp_list_node_p->next == NULL) {
        return NULL;
    }

    if (vp_list_node_p->next->item == item) {
        return vp_list_node_p; 
    }

    return vp_list_predecessor(vp_list_node_p->next, item); 
}

bool
vp_list_add_head(vp_list_t *vp_list_p, void *item)
{
    if (vp_list_p == NULL) {
        return false;
    }

    vp_list_node_t *node = (vp_list_node_t *)malloc(sizeof(vp_list_node_t));
    if (node == NULL) {
        return false;
    }

    node->next = NULL;
    node->item = item;
    
    if (vp_list_p->head == NULL && vp_list_p->tail == NULL) {
        vp_list_p->head = vp_list_p->tail = node;
    } else {
        node->next = vp_list_p->head;
        vp_list_p->head = node;
    }

    return true;
}

bool
vp_list_add_tail(vp_list_t *vp_list_p, void *item)
{
    if (vp_list_p == NULL) {
        return false;
    }

    vp_list_node_t *node = (vp_list_node_t *)malloc(sizeof(vp_list_node_t));
    if (node == NULL) {
        return false;
    }

    node->next = NULL;
    node->item = item;
 
    if (vp_list_p->head == NULL && vp_list_p->tail == NULL) {
        vp_list_p->head = vp_list_p->tail = node;
    } else {
        vp_list_p->tail->next = node;
        vp_list_p->tail = node;
    }

    return true;
}

vp_list_node_t *
vp_list_find_node(vp_list_node_t *node_p, void *item)
{
    if (node_p == NULL) {
        return NULL;
    }

    if (node_p->item == item) {
        return node_p;
    }

    return vp_list_find_node(node_p->next, item);
}

void *
vp_list_find(vp_list_t *vp_list_p, void *item)
{
    void *found_item = NULL;
    vp_list_node_t *node_p = vp_list_find_node(vp_list_p->head, item);

    if (node_p) {
        found_item = node_p->item; 
    }
    
    return found_item;
}

static void*
vp_list_remove_node(vp_list_t *vp_list_p, vp_list_node_t *target_node_p)
{
    if (target_node_p == NULL) {
        return NULL;
    }

    vp_list_node_t *after_node_p = target_node_p->next;
    vp_list_node_t *prev_node_p = vp_list_predecessor(vp_list_p->head, 
                                                target_node_p->item);
    
    if (prev_node_p) {
        prev_node_p->next = after_node_p;
    } else {
        vp_list_p->head = target_node_p->next;
    }

    if (after_node_p == NULL) {
        vp_list_p->tail = prev_node_p;
    }

    void* ret_item = target_node_p->item;
    free(target_node_p);

    return ret_item;
}

void *
vp_list_remove(vp_list_t *vp_list_p, void *item)
{
    vp_list_node_t *target_node_p = vp_list_find_node(vp_list_p->head, item);
    return vp_list_remove_node(vp_list_p, target_node_p);
}

void*
vp_list_remove_head(vp_list_t *vp_list_p)
{
    return vp_list_remove_node(vp_list_p, vp_list_p->head);
}

void*
vp_list_remove_tail(vp_list_t *vp_list_p)
{
    return vp_list_remove_node(vp_list_p, vp_list_p->tail);
}

static void
vp_list_apply_node(vp_list_node_t *node_p, vp_list_node_apply apply)
{
    if (node_p == NULL) {
        return;
    }

    apply(node_p->item);
    vp_list_apply_node(node_p->next, apply);
}

void
vp_list_apply(vp_list_t *vp_list_p, vp_list_node_apply apply)
{
    vp_list_apply_node(vp_list_p->head, apply);    
}

