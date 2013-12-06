#include <stdio.h>
#include "string.h"

static void 
swap(char *x, char *y) 
{
    char c = *x; 
    *x = *y; 
    *y = c;
}

void
string_permutations(char *str, int start_idx, int end_idx)
{
    int i = start_idx;

    if (start_idx == end_idx) {
        printf("%s\n", str);
        return;
    }

    while (i <= end_idx ) {
           swap(&str[i], &str[start_idx]);
           if (i != start_idx) {
                printf("%s\n",  str);
           }
           string_permutations(str, start_idx + 1, end_idx);
           swap(&str[i], &str[start_idx]);
           i++;
    }

}
