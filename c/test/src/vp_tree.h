#ifndef _VP_TREE_H_
#define _VP_TREE_H_

#include <stdbool.h>

typedef struct vp_tree_s vp_tree_t;
typedef int (*vp_tree_item_val_t)(void *item);
typedef void (*vp_tree_item_op_t)(void *item);

vp_tree_t* vp_tree_create(vp_tree_item_val_t item_val_func);
void vp_tree_free(vp_tree_t *vp_tree);

bool vp_tree_insert(vp_tree_t *vp_tree, void *item);
bool vp_tree_delete(vp_tree_t *vp_tree, void *item);

void* vp_tree_search(vp_tree_t *vp_tree, void *item);

void* vp_tree_min(vp_tree_t *vp_tree);
void* vp_tree_max(vp_tree_t *vp_tree);

void vp_tree_op_dfs(vp_tree_t *vp_tree, vp_tree_item_op_t vp_tree_item_op_func);
void vp_tree_op_bfs(vp_tree_t *vp_tree, vp_tree_item_op_t vp_tree_item_op_func);


#endif

