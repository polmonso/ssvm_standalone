/////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or       //
// modify it under the terms of the GNU General Public License         //
// version 2 as published by the Free Software Foundation.             //
//                                                                     //
// This program is distributed in the hope that it will be useful, but //
// WITHOUT ANY WARRANTY; without even the implied warranty of          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   //
// General Public License for more details.                            //
//                                                                     //
// Written and (C) by Aurelien Lucchi                                  //
// Contact <aurelien.lucchi@gmail.com> for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////
#ifndef PREDICT_H
#define PREDICT_H

//argp replacement
#ifdef _WIN32
#include "getopt.h"
#else
#include "unistd.h"
#include <getopt.h>
#endif


struct arguments
{
  bool export_all;
  char* image_dir;
  char* superpixel_labels;
  char* output_dir;
  char* weight_file;
  char* image_pattern;
  char* mask_dir;
  int nImages;
  int superpixelStepSize;
  int algo_type;
  char* config_file;
  char* overlay_dir;
  int dataset_type;
  bool use_average_vector;
};

//------------------------------------------------------------------------------
void print_usage();

static int parse_opt (int key, char *arg, struct arguments *argments);

//------------------------------------------------------------------------------

void create_average_weigth_vector(const char* input_dir, const char* output_file,
                                  const int start, const int end, const int step);

bool predict(int argc, char *argv[]);

#endif

