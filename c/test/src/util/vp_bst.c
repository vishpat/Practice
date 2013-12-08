#include "vp_bst.h"

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "vp_queue.h"

#define LEAF_NODE(node) ((node)->left == NULL && (node)->right == NULL)
#define SINGLE_CHILD_NODE(node) (((node)->left && (node)->right == NULL) ||\
                                 ((node)->left == NULL && (node)->right))

typedef struct vp_bst_node_s {
    void *item;
    struct vp_bst_node_s *parent;
    struct vp_bst_node_s *left;
    struct vp_bst_node_s *right;
} vp_bst_node_t;

struct vp_bst_s {
   vp_bst_node_t *root;
   vp_bst_item_val_t item_val;
};

vp_bst_t*
vp_bst_create(vp_bst_item_val_t item_val)
{
    vp_bst_t *vp_bst = (vp_bst_t *)malloc(sizeof(*vp_bst));
    
    if (vp_bst == NULL) {
        return NULL;
    }
    
    vp_bst->root = (vp_bst_node_t *)malloc(sizeof(*vp_bst->root));
    
    if (vp_bst->root == NULL) {
        free(vp_bst);
        return NULL;
    }

    vp_bst->root->parent = NULL;
    vp_bst->root->left = NULL;
    vp_bst->root->right = NULL;

    vp_bst->item_val = item_val;
    
    return vp_bst;
}

static void
vp_bst_free_node(vp_bst_node_t *node)
{
    vp_bst_node_t *left = node->left;
    vp_bst_node_t *right = node->right;

    if (left) {
        vp_bst_free_node(left);        
    }
    
    free(node);

    if (right) {
        vp_bst_free_node(right);
    }
}

void
vp_bst_free(vp_bst_t *vp_bst)
{
    vp_bst_free_node(vp_bst->root);
    free(vp_bst);
}

static vp_bst_node_t*
vp_bst_node_min(vp_bst_node_t *node)
{
    if (node->left == NULL) {
        return node;
    } else {
        return vp_bst_node_min(node->left);
    }
}

void*
vp_bst_min(vp_bst_t *vp_bst)
{
    vp_bst_node_t *min_node = vp_bst_node_min(vp_bst->root);
    return min_node ? min_node->item : NULL; 
}

static vp_bst_node_t*
vp_bst_node_max(vp_bst_node_t *node)
{
    if (node->right == NULL) {
        return node;
    } else {
        return vp_bst_node_max(node->right);
    }
}

void*
vp_bst_max(vp_bst_t *vp_bst)
{
    vp_bst_node_t *max_node = vp_bst_node_max(vp_bst->root);
    return max_node ? max_node->item : NULL; 
}


static vp_bst_node_t *
vp_bst_alloc_node(vp_bst_node_t *parent)
{
    vp_bst_node_t *node = (vp_bst_node_t *)malloc(sizeof(*node));
    
    if (node == NULL) {
        return node;
    }

    node->parent = parent;
    node->left = NULL;
    node->right = NULL;
    
    return node;
}

static bool
vp_bst_node_insert(vp_bst_t *vp_bst, vp_bst_node_t *node, void *item)
{
    int item_val = vp_bst->item_val(item);
    int node_val = vp_bst->item_val(node->item);
    bool status = false;

    if (vp_bst == NULL || node == NULL || item == NULL) {
        return false;
    }

    if (node_val == item_val) {
        return true;
    }

    if (item_val < node_val) {
        
        if (node->left) {
            return vp_bst_node_insert(vp_bst, node->left, item);
        } else {
            vp_bst_node_t *new_node = vp_bst_alloc_node(node);
            if (new_node) {
                new_node->item = item;
                node->left = new_node;
                new_node->parent = node;
                status = true;
            }
            return status;
        }
    }

    if (item_val > node_val) {
        if (node->right) {
            return vp_bst_node_insert(vp_bst, node->right, item);
        } else {
            vp_bst_node_t *new_node = vp_bst_alloc_node(node);
            if (new_node) {
                new_node->item = item;
                node->right = new_node;
                new_node->parent = node;
                status = true;
            }
            return status;
        }
    }

    return status;
}

bool
vp_bst_insert(vp_bst_t *vp_bst, void *item)
{
    if (vp_bst == NULL) {
        return false;
    }

    if (vp_bst->root->item == NULL) {
        vp_bst->root->item = item;
        return true;
    }

    return vp_bst_node_insert(vp_bst, vp_bst->root, item);
}

static vp_bst_node_t*
vp_bst_node_search(vp_bst_t *vp_bst, vp_bst_node_t *node, void *item)
{
    if (node == NULL || item == NULL) {
        return NULL;
    }

    if (vp_bst->item_val(node->item) == vp_bst->item_val(item)) {
        return node;
    }

    int item_val = vp_bst->item_val(item);
    int node_val = vp_bst->item_val(node->item);

    if (item_val < node_val) {
        return vp_bst_node_search(vp_bst, node->left, item);    
    } else {
        return vp_bst_node_search(vp_bst, node->right, item);
    }
}

void*
vp_bst_search(vp_bst_t *vp_bst, void *item)
{
    vp_bst_node_t *node = vp_bst_node_search(vp_bst, vp_bst->root, item);
    return node ? node->item : NULL;
}

static void
vp_bst_replace_child_node(vp_bst_node_t *old_child, 
                        vp_bst_node_t *new_child)
{
    vp_bst_node_t *parent = old_child->parent;
    assert(parent);

    if (parent->left == old_child) {
        parent->left = new_child;
    } else {
        parent->right = new_child;
    }

    if (new_child) {
        new_child->parent = parent;
    }
}

static vp_bst_node_t* 
vp_bst_delete_node_with_2_children(vp_bst_node_t *node)
{
    assert(node->right && node->left);

    vp_bst_node_t *parent = node->parent;
    vp_bst_node_t *replacement_node = vp_bst_node_max(node->left);

    if (LEAF_NODE(replacement_node)) {
        vp_bst_replace_child_node(replacement_node, NULL);
    } else if (SINGLE_CHILD_NODE(replacement_node)) {
        vp_bst_node_t *child = replacement_node->left ?
                    replacement_node->left : replacement_node->right;
        vp_bst_replace_child_node(replacement_node, child);
    }
            
    replacement_node->left = node->left;
    if (node->left) {
        node->left->parent = replacement_node;
    }

    replacement_node->right = node->right;
    if (node->right) {
        node->right->parent = replacement_node;
    }

    replacement_node->parent = parent;

    if (parent) {
        vp_bst_replace_child_node(node, replacement_node);        
    }

    return replacement_node;
}

bool
vp_bst_delete(vp_bst_t *vp_bst, void *item)
{
   vp_bst_node_t *node = vp_bst_node_search(vp_bst, vp_bst->root, item);

   if (node == NULL) {
       return false;
   }

   if (node == vp_bst->root) {
        
        if (LEAF_NODE(node)) { 
            vp_bst->root = NULL;
        } else if (SINGLE_CHILD_NODE(node)) {
            vp_bst_node_t *child = node->left ? node->left : node->right;
            vp_bst->root = child;  
            child->parent = NULL;
        } else {
            vp_bst->root = vp_bst_delete_node_with_2_children(vp_bst->root);
        }
    } else {
        
        if (LEAF_NODE(node)) {
            vp_bst_replace_child_node(node, NULL);
        } else if (SINGLE_CHILD_NODE(node)) {
            vp_bst_node_t *child = node->left ? node->left : node->right;
            vp_bst_replace_child_node(node, child);
        } else {
            (void)vp_bst_delete_node_with_2_children(node);
        }    
    }

    free(node);

    return true;
}

static void
vp_bst_node_op_dfs(vp_bst_node_t *node, vp_bst_item_op_t vp_bst_item_op_func)
{
    if (node == NULL) {
        return;
    }

    vp_bst_node_op_dfs(node->left, vp_bst_item_op_func);
    vp_bst_item_op_func(node->item);
    vp_bst_node_op_dfs(node->right, vp_bst_item_op_func);
}

void
vp_bst_op_dfs(vp_bst_t *vp_bst, vp_bst_item_op_t vp_bst_item_op_func)
{
    return vp_bst_node_op_dfs(vp_bst->root, vp_bst_item_op_func);    
}

void
vp_bst_op_bfs(vp_bst_t *vp_bst, vp_bst_item_op_t vp_bst_item_op_func)
{
    vp_queue_t *q = vp_queue_create();

    if (q == NULL) {
        return;
    }
    
    if (vp_bst->root) {
        vp_queue_enqueue(q, vp_bst->root);
    }

    while (!vp_queue_empty(q)) {
        vp_bst_node_t *node = (vp_bst_node_t *)vp_queue_dequeue(q);
        
        vp_bst_item_op_func(node->item);

        if (node->left) {
            vp_queue_enqueue(q, node->left);
        }

        if (node->right) {
            vp_queue_enqueue(q, node->right);
        }
    }

    vp_queue_free(q);
}
