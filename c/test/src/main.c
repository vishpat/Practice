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

    return 0;
}

