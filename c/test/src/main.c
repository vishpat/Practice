#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "vp_string.h"
#include "vp_misc.h"
#include "vp_list.h"
#include "vp_bst.h"

int cmp(const void *x, const void *y) 
{
    return *((int *)x) - *((int *)y);
}

int 
int_val(const void *v)
{
    return *((int *)v);    
}

void
print_int_val(const void *v)
{
    printf("%d\n", *((int *)v));       
}

#define NUM_CNT 10 
int
main(int argc, char ** argv) 
{
    int nums[NUM_CNT];
    int i;

    for (i = 0; i < NUM_CNT; i++) {
        nums[i] = random() % 4096;
    }

    qsort(nums, NUM_CNT, sizeof(nums[0]), cmp);
    for (i = 0; i < NUM_CNT; i++) {
        printf("%d\n", nums[i]);
    }

    vp_bst_t *bst = vp_bst_create(int_val);
    for (i = 0; i < NUM_CNT; i++) {
       vp_bst_insert(bst, &nums[i]); 
    }

    vp_bst_op_bfs(bst, print_int_val); 

    return 0;
}

