#include <string.h>
#include "string.h"

void
swap(char *x, char *y)
{
    char c = *x;
    *x = *y;
    *y = c;
}

int
main(int argc, char ** argv) {
    char s[] = {'a', 'b', 'c', 'd', 0};
    string_permutations(s, 0, strlen(s) - 1);
    return 0;
}

