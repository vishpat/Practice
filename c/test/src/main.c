#include <string.h>
#include <stdio.h>
#include "vp_string.h"
#include "vp_misc.h"
#include "vp_list.h"

void
print_list_item(void *item)
{
    printf("%d\n", *((int *)item));
}

int
main(int argc, char ** argv) {
    int items[] = {1, 2, 4, 5, 6, 8, 9};
    int i = 0;
    
    vp_list_t *list = vp_list_create();
    
    for (i = 0; i < sizeof(items)/sizeof(items[0]); i++) {
        vp_list_add_head(list, &items[i]);
    }
    
    vp_list_apply(list, print_list_item);
    vp_list_free(list, NULL);
}

