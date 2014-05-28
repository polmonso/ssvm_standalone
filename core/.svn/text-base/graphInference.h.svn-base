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

#ifndef GRAPH_INFERENCE_H
#define GRAPH_INFERENCE_H

#include <stdio.h>

#include "Feature.h"
#include "Slice_P.h"
#include "globalsE.h"
#include "oSVM.h"

#include "inference_globals.h"
#include "energyParam.h"

//------------------------------------------------------------------------------

#define T_GI_LIBDAI 0
#define T_GI_MRF 1
#define T_GI_QPBO 2
//#define T_GI_FASTPD 3
#define T_GI_MAX 4
#define T_GI_MAXFLOW 5
#define T_GI_OPENGM 6
#define T_GI_SAMPLING 7
#define T_GI_ICM 8
#define T_GI_LIBDAI_ICM 9
#define T_GI_LIBDAI_ICM_QPBO 10
#define T_GI_MF 11
#define T_GI_MULTIOBJ 12

//------------------------------------------------------------------------------

class GraphInference
{
 public:

  GraphInference() {}
  GraphInference(Slice_P* _slice, 
                 const EnergyParam* _param,
                 double* _smw,
                 Feature* _feature,
                 map<sidType, nodeCoeffType>* _nodeCoeffs,
                 map<sidType, edgeCoeffType>* _edgeCoeffs
                 );

  virtual ~GraphInference();

  virtual void addLocalEdges() { return; }

  virtual void addUnaryNodes(labelType* globalNodeLabels = 0) { return; }

  /**
   * Compute energy for a given configuration nodeLabels
   */
  double computeEnergy(labelType* nodeLabels);

  void computeNodePotentials(double**& unaryPotentials, double& maxPotential);

  void init();

  virtual double run(labelType* inferredLabels,
                     int id,
                     size_t maxiter,
                     labelType* nodeLabelsGroundTruth = 0,
                     bool computeEnergyAtEachIteration = false,
                     double* _loss = 0) { assert(0); return 0; }

  static void setColormap(map<ulong, labelType>& _classIdxToLabel) { classIdxToLabel = _classIdxToLabel;}

  Slice_P* slice;
  double* smw;

  const EnergyParam* param;
  double* lossPerLabel;
  labelType* groundTruthLabels;   /* ground truth labels */
  Feature* feature;

  // compute pairwise potential for 2 given nodes s and sn
  inline double computePairwisePotential(Slice_P* slice, supernode* s,
                                         supernode* sn,
                                         labelType s_label,
                                         labelType sn_label);

  // compute pairwise potential for 2 given nodes s and sn
  // distance adaptive pairwise term
  inline double computePairwisePotential_distance(Slice_P* slice, supernode* s,
                                                  supernode* sn,
                                                  labelType s_label,
                                                  labelType sn_label);

  /**
   * Assume features were precomputed with the function precomputeFeatures
   */
  inline double computeUnaryPotential(Slice_P* slice, sidType sid,
                                      labelType label) {
    double p = 0;
    osvm_node *n = slice->getFeature(sid);


#if DEBUG_MSRC

#if USE_SPARSE_VECTORS

    //static int ncall = 0;

    if(label < 21) {
      // code to weight combination of features + SVM predictions
      // use one weight per feature dimension, except for SVM predictions
      // for which we use one weight per scale.
      int fvSize = slice->getFeatureSize(sid);
      int fvSize_predictions = 21*param->nScales;
      int last_feature_idx = fvSize - fvSize_predictions;
      int inc = n[0].index;
      int widx = SVM_FEAT_INDEX(param, label, inc - 1);
      //if(sid == 100 && ncall > 16)
      //  printf("[graphInference] ncall %d s 0 label %d %d %d\n", ncall, label, inc, widx, n[0].value);
      for(int s = 0; s < last_feature_idx; s++) {
        p += n[s].value*smw[widx];
        //if(isinf(p) || isnan(p)) {
        //  printf("[graphInference] computeUnary %d %d %g %g\n", s, widx, n[s].value,smw[widx]);
        //}
        //if(sid == 100 && ncall > 16)
        //  printf("[graphInference] ncall %d s %d label %d %d %d %d %d %g %g %g\n", ncall, s, label, n[s+1].index, n[s].index, inc, widx, n[s].value, smw[widx], p);
        inc = n[s+1].index - n[s].index;
        widx += inc*SVM_FEAT_NUM_CLASSES(param);
      }

      /*
      if(sid == 100 && ncall > 16) {
        printf("[graphInference] ncall %d label %d %g\n", ncall, label, p);
        int s = fvSize - (21*1);
        printf("[graphInference] ncall %d label %d %d %d\n", ncall, label, fvSize, s);
      }
      */

      // add svm prediction
      int fvSize_full = slice->getFeatureSize();
      for(int l = 1; l <= param->nScales; ++l) {
        int s = fvSize - (21*l) + label;
        int w_idx = SVM_FEAT_INDEX(param, 0, fvSize_full - (21*l));
        p += n[s].value*smw[w_idx];
        //if(sid == 100 && ncall > 16)
        //  printf("[graphInference] ncall %d s2 %d label %d %d %d %g %g %g\n", ncall, s, label, w_idx, fvSize_full, n[s].value, smw[w_idx], p);
      }

    }

#else

    if(label < 21) {
      // code to weight combination of features + SVM predictions
      // use one weight per feature dimension, except for SVM predictions
      // for which we use one weight per scale.
      int fvSize = slice->getFeatureSize();
      int widx = SVM_FEAT_INDEX(param, label, 0);
      int fvSize_predictions = 21*param->nScales;
      //printf("fvSize %d %d\n", fvSize, fvSize_predictions);
      for(int s = 0; s < fvSize - fvSize_predictions; s++) {
        p += n[s].value*smw[widx];
        if(isinf(p) || isnan(p)) {
          printf("[graphInference] computeUnary %d %d %g %g\n", s, widx, n[s].value,smw[widx]);
          exit(-1);
        }
        widx += SVM_FEAT_NUM_CLASSES(param);
      }
      for(int l = 1; l <= param->nScales; ++l) {
        //printf("l %d\n", l);
        int s = fvSize - (21*l) + label;
        int w_idx = SVM_FEAT_INDEX(param, 0, fvSize - (21*l));
        p += n[s].value*smw[w_idx];
        //if(isinf(p) || isnan(p)) {
        //  printf("[graphInference] computeUnary %d %d %g %g\n", s, widx, n[s].value,smw[widx]);
        //}
      }
    }

#endif

#ifdef W_OFFSET
    p += smw[label];
#endif

#else

#if USE_SPARSE_VECTORS
    assert(0);
    // To be implemented...
#endif

    int fidx = 0;
    int widx = SVM_FEAT_INDEX(param, label, fidx);
    while(n[fidx].index != -1) {
      p += n[fidx].value*smw[widx];
      if(isinf(p) || isnan(p)) {
        printf("[graphInference] computeUnary image (%ld, %s) sid %d label %d -> %d %d %d %g %g\n", slice->getId(), slice->getName().c_str(), sid, (int) label, fidx, widx, n[fidx].index, n[fidx].value,smw[widx]);
        oSVM::print(n);
        exit(-1);
      }
      fidx++;
      widx += SVM_FEAT_NUM_CLASSES(param);
    }

#ifdef W_OFFSET
    p += smw[label];
#endif

#endif

    return p;
  }

  inline double computeUnaryPotential_copy(Slice_P* slice, sidType sid,
                                           labelType label, osvm_node *n) {
    double p = 0;
    feature->getFeatureVector(n, slice, sid);

    int fidx = 0;
    while(n[fidx].index != -1) {
      p += n[fidx].value*smw[SVM_FEAT_INDEX(param,label,fidx)];
      fidx++;
    }
#ifdef W_OFFSET
    p += smw[label];
#endif
    return p;
  }

 protected:

  map<sidType, nodeCoeffType>* nodeCoeffs;
  map<sidType, edgeCoeffType>* edgeCoeffs;

  // ugly hack to remove void labels
  static map<ulong, labelType> classIdxToLabel;

};

double GraphInference::computePairwisePotential(Slice_P* slice, supernode* s,
                                                supernode* sn,
                                                labelType s_label,
                                                labelType sn_label)
{
  int idx;
  double energy = 0;
  int p = 0;
  if(s->id < sn->id) {
    p = (sn_label*param->nClasses) + s_label;
  } else {
    p = (s_label*param->nClasses) + sn_label;
  }

  int gradientIdx = slice->getGradientIdx(s->id, sn->id);
  int orientationIdx = slice->getOrientationIdx(s->id, sn->id);

  p += (orientationIdx*param->nClasses*param->nClasses);

  // w[0..nClasses-1] contains the unary weights
  for(int g = 0; g <= gradientIdx; g++) {
    idx = (g*param->nClasses*param->nClasses*param->nOrientations) + p;
    energy += smw[idx+param->nUnaryWeights]; // nUnaryWeights is the offset due to unary terms
  }
  return energy;
}


double GraphInference::computePairwisePotential_distance(Slice_P* slice, supernode* s,
                                                         supernode* sn,
                                                         labelType s_label,
                                                         labelType sn_label)
{

#if !USE_LONG_RANGE_EDGES
  assert(0);
#endif

  int idx;
  double energy = 0;
  int p = 0;
  if(s->id < sn->id) {
    p = (sn_label*param->nClasses) + s_label;
  } else {
    p = (s_label*param->nClasses) + sn_label;
  }

  int gradientIdx = slice->getGradientIdx(s->id, sn->id);
  int orientationIdx = slice->getOrientationIdx(s->id, sn->id); 
  int distanceIdx = slice->getDistanceIdx(s->id, sn->id);

  p += (orientationIdx*param->nClasses*param->nClasses);

  p += distanceIdx*param->nGradientLevels*param->nClasses*param->nClasses*param->nOrientations;

  // w[0..nClasses-1] contains the unary weights
  for(int g = 0; g <= gradientIdx; g++) {
    idx = (g*param->nClasses*param->nClasses*param->nOrientations) + p;
    energy += smw[idx+param->nUnaryWeights]; // nUnaryWeights is the offset due to unary terms
  }
  return energy;
}

#endif //GRAPH_INFERENCE_H
