
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#include "train.h"

int main (int argc, char* argv[])
{
  if(argc < 2){
    fprintf(stderr, "Missing configuration file argument\n");
    fprintf(stderr, "Usage: train configuration_file\n");
    return EXIT_FAILURE;
  }
  train(argv[1]);
  return EXIT_SUCCESS;
}
