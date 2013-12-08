#ifndef _VP_BST_H_
#define _VP_BST_H_

#include <stdbool.h>

typedef struct vp_bst_s vp_bst_t;
typedef int (*vp_bst_item_val_t)(void *item);
typedef void (*vp_bst_item_op_t)(void *item);

vp_bst_t* vp_bst_create(vp_bst_item_val_t item_val_func);
void vp_bst_free(vp_bst_t *vp_bst);

bool vp_bst_insert(vp_bst_t *vp_bst, void *item);
bool vp_bst_delete(vp_bst_t *vp_bst, void *item);

void* vp_bst_search(vp_bst_t *vp_bst, void *item);

void* vp_bst_min(vp_bst_t *vp_bst);
void* vp_bst_max(vp_bst_t *vp_bst);

void vp_bst_op_dfs(vp_bst_t *vp_bst, vp_bst_item_op_t vp_bst_item_op_func);
void vp_bst_op_bfs(vp_bst_t *vp_bst, vp_bst_item_op_t vp_bst_item_op_func);


#endif

