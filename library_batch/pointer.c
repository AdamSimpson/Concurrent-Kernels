#include "stdio.h"

void setp_(void** pointer)
{
     printf("pointer in C before null %p\n", *pointer);
     *pointer = NULL;
     printf("pointer in C after null %p\n", *pointer);
     printf("**pointer after null %p\n", pointer)
}

void printp_(void* pointer)
{
    printf("pointer: %p\n", pointer);
}
