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

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

// SliceMe
#include "Config.h"
#include "Slice.h"
#include "colormap.h"
#include "utils.h"
#include "globalsE.h"
#include "globals.h"
#include "inference.h"
#include "Feature.h"
#include "F_Combo.h"

#include "gi_libDAI.h"

#include "energyParam.h"
#include "svm_struct_api_types.h"
#include "svm_struct_api.h"

using namespace std;

/* Variables for the arguments.*/
const char *argp_program_version =
  "predict 0.1";
const char *argp_program_bug_address =
  "<aurelien.lucchi@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "Inference";

/* A description of the arguments we accept. */
static char args_doc[] = "args";

/* Program options */
static struct argp_option options[] = {
  {"all",'a',  "all", 0, "export marginals and also run inference using unary potentials only (useful for debugging)"},
  {"config_file",'c',  "config_file",0, "config_file"},
  {"image_dir",'i',  "image_dir",0, "input directory"},
  {"algo_type",'g',  "algo_type",0, "algo_type"},
  {"image_pattern",'k',  "image_pattern",0, "image_pattern"},
  {"superpixel_labels",'l',  "superpixel_labels",0, "path of the file containing the labels for the superpixels (labels have be ordered by rows)"},
  {"mask_dir",'m',  "mask_dir",0, "mask directory"},
  {"nImages",'n',  "nImages", 0, "number of images to process"},
  {"output_dir",'o',  "output_dir",0, "output filename"},
  {"superpixelStepSize",'s',  "superpixelStepSize",0,"superpixel step size"},
  {"dataset_type",'t',  "dataset_type",0, "type (0=training, 1=test)"},
  {"verbose",'v',  "verbose",0, "verbose"},
  {"weight_file",'w',  "weight_file",0, "weight_file"},
  {"overlay",'y',  "overlay",0, "overlay directory"},
  { 0 }
};


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
};

arguments args;


//------------------------------------------------------------------------------

/* Parse a single option. */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
  struct arguments *argments = (arguments*)state->input;

  switch (key)
    {
    case 'a':
      if(arg && arg[0] == '1') {
        argments->export_all = true;
      } else {
        argments->export_all = false;
      }
      break;
    case 'c':
      argments->config_file = arg;
      break;
    case 'g':
      if(arg!=0)
        argments->algo_type = atoi(arg);
      break;
    case 'i':
      argments->image_dir = arg;
      break;
    case 'k':
      if(arg!=0)
        argments->image_pattern = arg;
      break;
    case 'l':
      argments->superpixel_labels = arg;
      break;
    case 'm':
      argments->mask_dir = arg;
      break;
    case 'n':
      if(arg!=0)
        argments->nImages = atoi(arg);
      break;
    case 'o':
      argments->output_dir = arg;
      break;
    case 's':
      argments->superpixelStepSize = atoi(arg);
      break;
    case 't':
      if(arg!=0)
        argments->dataset_type = atoi(arg);
      break;
    case 'v':
      if(arg && arg[0] == '1')
        verbose = true;
      else
        verbose = false;
      break;
    case 'w':
      argments->weight_file = arg;
      break;
    case 'y':
      argments->overlay_dir = arg;
      break;

    case ARGP_KEY_ARG:
      //if (state->arg_num >= 2)
      /* Too many arguments. */
      argp_usage (state);
      break;

    case ARGP_KEY_END:
      /*
      // Not enough arguments
      if (state->arg_num < 1)
      argp_usage (state);
      */
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };

//------------------------------------------------------------------------------

int main(int argc,char* argv[])
{
  args.image_dir = 0;
  args.mask_dir = (char*)"";
  args.nImages = -1;
  args.superpixel_labels = 0;
  args.output_dir = (char*)"./inference/";
  args.algo_type = T_GI_LIBDAI;
  args.weight_file = 0;
  verbose = false;
  args.image_pattern = (char*)"png";
  args.superpixelStepSize = SUPERPIXEL_DEFAULT_STEP_SIZE;
  args.config_file = 0;
  args.overlay_dir = 0;
  args.export_all = false;
  args.dataset_type = 0;
  const bool compress_image = false;

  argp_parse (&argp, argc, argv, 0, 0, &args);
  
  if(args.overlay_dir == 0) {
    args.overlay_dir = args.output_dir;
  }

  // config file is loaded in read_struct_examples
  char * ppPath = getenv ("LOCALHOME");
  if(ppPath == 0) {
    ppPath = getenv ("HOME");
  }
  string pPath(ppPath);

  string config_tmp;
  Config* config = new Config(args.config_file);
  Config::setInstance(config);

  set_default_parameters(config);

  mkdir(args.output_dir, 0777);

  string imageDir;
  string maskDir;
  if(args.image_dir != 0) {
    imageDir = args.image_dir;
  } else {
    if(args.dataset_type == 0) {
      Config::Instance()->getParameter("trainingDir", imageDir);
      Config::Instance()->getParameter("maskTrainingDir", maskDir);
    } else {
      Config::Instance()->getParameter("testDir", imageDir);
      Config::Instance()->getParameter("maskTestDir", maskDir);
    }
    if(!isDirectory(imageDir) && !fileExists(imageDir)) {
      imageDir = pPath + imageDir;
    }
  }

  // mask directory
  if(!isDirectory(maskDir) && !maskDir.empty()) {
    maskDir = pPath + maskDir;
  }

  vector<eFeatureType> feature_types;
  int paramFeatureTypes = 0;
  if(config->getParameter("featureTypes", config_tmp)) {
    paramFeatureTypes = atoi(config_tmp.c_str());
    getFeatureTypes(paramFeatureTypes, feature_types);
  }

  EnergyParam param(args.weight_file);

  // TODO: Find a better way to change the labels!
  if(param.nClasses == 3) {
    printf("[svm_struct] Set class labels\n");
    BACKGROUND = 0;
    BOUNDARY = 1;
    FOREGROUND = 2;
  }

  Slice_P* slice = 0;
  Feature* feature = 0;
  int featureSize = 0;
  loadDataAndFeatures(imageDir, maskDir, config, slice, feature, &featureSize);

  string colormapFilename;
  getColormapName(colormapFilename);
  printf("[Main] Colormap=%s\n", colormapFilename.c_str());
  map<labelType, ulong> labelToClassIdx;
  getLabelToClassMap(colormapFilename.c_str(), labelToClassIdx);

  labelType* groundTruthLabels = 0;
  double* lossPerLabel = 0;
  const int nExamples = 1;
  const string score_filename = "";
  for(int sid = 0; sid < nExamples; sid++) {

    SPATTERN p;
    p.id = 0;
    p.slice = slice;
    p.feature = feature;

    /*
    segmentImage(p,
                 args.output_dir,
                 args.algo_type,
                 args.weight_file,
                 &labelToClassIdx,
                 score_filename,
                 groundTruthLabels, lossPerLabel,
                 compress_image,
                 args.overlay_dir,
                 METRIC_SUPERNODE_BASED_01);
    */

    GI_libDAI* gi_Inference = new GI_libDAI(slice,
                                            &param,
                                            param.weights,
                                            groundTruthLabels,
                                            lossPerLabel,
                                            feature,
                                            0, 0);

    ulong nNodes = slice->getNbSupernodes();

    size_t maxiter = 100;
    labelType* nodeLabels = new labelType[nNodes];
    double lambda = 0;

    // Run inference to get first mode
    stringstream soutColoredImage;
    soutColoredImage << args.output_dir << "/";
    soutColoredImage << getNameFromPathWithoutExtension(slice->getName());
    soutColoredImage << ".png";

    printf("Running inference for MAP: %s\n", soutColoredImage.str().c_str());
    double energy = gi_Inference->run(nodeLabels, 0, maxiter);
    printf("energy %g\n", energy);
    lambda = 0.1*(fabs(energy) / nNodes);
      
    // output image
    printf("Exporting %s\n", soutColoredImage.str().c_str());
    slice->exportSupernodeLabels(soutColoredImage.str().c_str(),
                                 param.nClasses,
                                 nodeLabels,
                                 nNodes,
                                 &labelToClassIdx);

    // Get (i+1)-th modes
    labelType* nodeLabels_mode = new labelType[nNodes];
    for(int i = 0; i < 5; ++i) {

      printf("Round %d\n", i);      

      gi_Inference->allocateGraph();
      gi_Inference->precomputePotentials();
      dai::Real** unaryPotentials = gi_Inference->getUnaryPotentials();
      assert(unaryPotentials != 0);
      
      for(ulong n = 0; n < nNodes; ++n) {
        //printf("%ld:%d\n", n, nodeLabels[n]);
        unaryPotentials[n][nodeLabels[n]] -= (i+1)*lambda; 
      }
      gi_Inference->addNodes();
      
      stringstream soutColoredImage;
      soutColoredImage << args.output_dir << "/";
      soutColoredImage << getNameFromPathWithoutExtension(slice->getName());
      soutColoredImage << "_";
      soutColoredImage << i;
      soutColoredImage << ".png";

      printf("Running inference for round %d: %s\n", i, soutColoredImage.str().c_str());
      double energy = gi_Inference->run(nodeLabels_mode, 0, maxiter);
      printf("energy %g\n", energy);
      lambda = 0.1*(fabs(energy) / nNodes);
      
      // output image
      printf("Exporting %s\n", soutColoredImage.str().c_str());
      slice->exportSupernodeLabels(soutColoredImage.str().c_str(),
                                   param.nClasses,
                                   nodeLabels_mode,
                                   nNodes,
                                   &labelToClassIdx);
      
    }

    delete gi_Inference;
    delete[] nodeLabels;      

  }

  printf("[Main] Cleaning\n");
  printf("[Main] Done\n");
  return 0;
}
