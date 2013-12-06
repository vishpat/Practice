#include <string.h>
#include "vp_string.h"

int
main(int argc, char ** argv) {
    char s[] = {'a', 'b', 'c', 'd', 0};
    vp_string_permutations(s, 0, strlen(s) - 1);
    return 0;
}

