#include "vp_tree.h"

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "vp_queue.h"

#define LEAF_NODE(node) ((node)->left == NULL && (node)->right == NULL)
#define SINGLE_CHILD_NODE(node) (((node)->left && (node)->right == NULL) ||\
                                 ((node)->left == NULL && (node)->right))

typedef struct vp_tree_node_s {
    void *item;
    struct vp_tree_node_s *parent;
    struct vp_tree_node_s *left;
    struct vp_tree_node_s *right;
} vp_tree_node_t;

struct vp_tree_s {
   vp_tree_node_t *root;
   vp_tree_item_val_t item_val;
};

vp_tree_t*
vp_tree_create(vp_tree_item_val_t item_val)
{
    vp_tree_t *vp_tree = (vp_tree_t *)malloc(sizeof(*vp_tree));
    
    if (vp_tree == NULL) {
        return NULL;
    }
    
    vp_tree->root = (vp_tree_node_t *)malloc(sizeof(*vp_tree->root));
    
    if (vp_tree->root == NULL) {
        free(vp_tree);
        return NULL;
    }

    vp_tree->root->parent = NULL;
    vp_tree->root->left = NULL;
    vp_tree->root->right = NULL;

    vp_tree->item_val = item_val;
    
    return vp_tree;
}

void
vp_tree_free(vp_tree_t *vp_tree)
{
    free(vp_tree);
}

static vp_tree_node_t*
vp_tree_node_min(vp_tree_node_t *node)
{
    if (node->left == NULL) {
        return node;
    } else {
        return vp_tree_node_min(node->left);
    }
}

void*
vp_tree_min(vp_tree_t *vp_tree)
{
    vp_tree_node_t *min_node = vp_tree_node_min(vp_tree->root);
    return min_node ? min_node->item : NULL; 
}

static vp_tree_node_t*
vp_tree_node_max(vp_tree_node_t *node)
{
    if (node->right == NULL) {
        return node;
    } else {
        return vp_tree_node_max(node->right);
    }
}

void*
vp_tree_max(vp_tree_t *vp_tree)
{
    vp_tree_node_t *max_node = vp_tree_node_max(vp_tree->root);
    return max_node ? max_node->item : NULL; 
}


static vp_tree_node_t *
vp_tree_alloc_node(vp_tree_node_t *parent)
{
    vp_tree_node_t *node = (vp_tree_node_t *)malloc(sizeof(*node));
    
    if (node == NULL) {
        return node;
    }

    node->parent = parent;
    node->left = NULL;
    node->right = NULL;
    
    return node;
}

static bool
vp_tree_node_insert(vp_tree_t *vp_tree, vp_tree_node_t *node, void *item)
{
    int item_val = vp_tree->item_val(item);
    int node_val = vp_tree->item_val(node->item);
    bool status = false;

    if (vp_tree == NULL || node == NULL || item == NULL) {
        return false;
    }

    if (node_val == item_val) {
        return true;
    }

    if (item_val < node_val) {
        
        if (node->left) {
            return vp_tree_node_insert(vp_tree, node->left, item);
        } else {
            vp_tree_node_t *new_node = vp_tree_alloc_node(node);
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
            return vp_tree_node_insert(vp_tree, node->right, item);
        } else {
            vp_tree_node_t *new_node = vp_tree_alloc_node(node);
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
vp_tree_insert(vp_tree_t *vp_tree, void *item)
{
    if (vp_tree == NULL) {
        return false;
    }

    if (vp_tree->root->item == NULL) {
        vp_tree->root->item = item;
        return true;
    }

    return vp_tree_node_insert(vp_tree, vp_tree->root, item);
}

static vp_tree_node_t*
vp_tree_node_search(vp_tree_t *vp_tree, vp_tree_node_t *node, void *item)
{
    if (node == NULL || item == NULL) {
        return NULL;
    }

    if (vp_tree->item_val(node->item) == vp_tree->item_val(item)) {
        return node;
    }

    int item_val = vp_tree->item_val(item);
    int node_val = vp_tree->item_val(node->item);

    if (item_val < node_val) {
        return vp_tree_node_search(vp_tree, node->left, item);    
    } else {
        return vp_tree_node_search(vp_tree, node->right, item);
    }
}

void*
vp_tree_search(vp_tree_t *vp_tree, void *item)
{
    vp_tree_node_t *node = vp_tree_node_search(vp_tree, vp_tree->root, item);
    return node ? node->item : NULL;
}

static void
vp_tree_replace_child_node(vp_tree_node_t *old_child, 
                        vp_tree_node_t *new_child)
{
    vp_tree_node_t *parent = old_child->parent;
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

static vp_tree_node_t* 
vp_tree_delete_node_with_2_children(vp_tree_node_t *node)
{
    assert(node->right && node->left);

    vp_tree_node_t *parent = node->parent;
    vp_tree_node_t *replacement_node = vp_tree_node_max(node->left);

    if (LEAF_NODE(replacement_node)) {
        vp_tree_replace_child_node(replacement_node, NULL);
    } else if (SINGLE_CHILD_NODE(replacement_node)) {
        vp_tree_node_t *child = replacement_node->left ?
                    replacement_node->left : replacement_node->right;
        vp_tree_replace_child_node(replacement_node, child);
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
        vp_tree_replace_child_node(node, replacement_node);        
    }

    return replacement_node;
}

bool
vp_tree_delete(vp_tree_t *vp_tree, void *item)
{
   vp_tree_node_t *node = vp_tree_node_search(vp_tree, vp_tree->root, item);

   if (node == NULL) {
       return false;
   }

   if (node == vp_tree->root) {
        
        if (LEAF_NODE(node)) { 
            vp_tree->root = NULL;
        } else if (SINGLE_CHILD_NODE(node)) {
            vp_tree_node_t *child = node->left ? node->left : node->right;
            vp_tree->root = child;  
            child->parent = NULL;
        } else {
            vp_tree->root = vp_tree_delete_node_with_2_children(vp_tree->root);
        }
    } else {
        
        if (LEAF_NODE(node)) {
            vp_tree_replace_child_node(node, NULL);
        } else if (SINGLE_CHILD_NODE(node)) {
            vp_tree_node_t *child = node->left ? node->left : node->right;
            vp_tree_replace_child_node(node, child);
        } else {
            (void)vp_tree_delete_node_with_2_children(node);
        }    
    }

    free(node);

    return true;
}

static void
vp_tree_node_op_dfs(vp_tree_node_t *node, vp_tree_item_op_t vp_tree_item_op_func)
{
    if (node == NULL) {
        return;
    }

    vp_tree_node_op_dfs(node->left, vp_tree_item_op_func);
    vp_tree_item_op_func(node->item);
    vp_tree_node_op_dfs(node->right, vp_tree_item_op_func);
}

void
vp_tree_op_dfs(vp_tree_t *vp_tree, vp_tree_item_op_t vp_tree_item_op_func)
{
    return vp_tree_node_op_dfs(vp_tree->root, vp_tree_item_op_func);    
}

void
vp_tree_op_bfs(vp_tree_t *vp_tree, vp_tree_item_op_t vp_tree_item_op_func)
{
    vp_queue_t *q = vp_queue_create();

    if (q == NULL) {
        return;
    }
    
    if (vp_tree->root) {
        vp_queue_enqueue(q, vp_tree->root);
    }

    while (!vp_queue_empty(q)) {
        vp_tree_node_t *node = (vp_tree_node_t *)vp_queue_dequeue(q);
        
        vp_tree_item_op_func(node->item);

        if (node->left) {
            vp_queue_enqueue(q, node->left);
        }

        if (node->right) {
            vp_queue_enqueue(q, node->right);
        }
    }
}
