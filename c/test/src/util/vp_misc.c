#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "vp_misc.h"

int 
vp_binomial_coefficient(unsigned int n, unsigned int k)
{
    int arr_size = (n + 1)*(k + 1)*sizeof(int);
    int *arr = (int *)malloc(arr_size);
    if (arr == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return -1;
    }

    memset(arr, 0, arr_size);

    int **c = (int **)malloc((n + 1)*sizeof(int *));
    if (c == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return -1;
    }
   
    int i, j, c1, c2;
    for (i = 0; i < n + 1; i++) {
        c[i] = &arr[i*(k + 1)];
        for (j = 0; j < k + 1; j++) {
            if (j == 0) {
                c[i][j] = 1;
            } else if (i == 0) {
                c[i][j] = 0;
            } else {
                c1 = c[i - 1][j]; 
                c2 = c[i - 1][j - 1];
                c[i][j] = c1 + c2;
            }
       }
    }
    
    int ret = c[n][k];

    free(c);
    free(arr);

    return ret;
}
