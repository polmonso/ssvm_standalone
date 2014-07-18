#include "predict.h"
#include "unistd.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {

    if(predict(argc, argv) == false)
        exit(EXIT_FAILURE);

    printf("[Main] Cleaning\n");
    printf("[Main] Done\n");
    return 0;
}

