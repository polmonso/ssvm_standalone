/***********************************************************************/
/*                                                                     */
/*   svm_struct_learn_custom.c (instantiated for SVM-perform)          */
/*                                                                     */
/*   Allows implementing a custom/alternate algorithm for solving      */
/*   the structual SVM optimization problem. The algorithm can use     */ 
/*   full access to the SVM-struct API and to SVM-light.               */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 09.01.08                                                    */
/*                                                                     */
/*   Copyright (c) 2008  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

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

#include <iomanip>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include <sys/types.h>
#ifdef _WIN32
#include <io.h>
#include <direct.h>
#define mkdir(x,y) _mkdir(x)
#include <time.h>
#include <winsock2.h>
#include "gettimeofday.h"
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#include "constraint_set.h"
#include "label_cache.h"
#include "svm_struct_learn_custom.h"
#include "svm_struct_api.h"
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
#include "svm_struct/svm_struct_learn.h"
#include "svm_struct_globals.h"

#include "Config.h"

#include "highgui.h"

#include "energyParam.h"
#include "graphInference.h"
#include "inference.h"

//------------------------------------------------------------------------MACROS

#define BUFFER_SIZE 250

// If greater than 1, output dscore, norm(dfy), loss
// If greater than 2, output dfy
#define CUSTOM_VERBOSITY 3

#define CUSTOM_VERBOSITY_F(X, Y) if(CUSTOM_VERBOSITY > X) { Y }

//---------------------------------------------------------------------FUNCTIONS

void write_vector(const char* filename, double* v, int size_v)
{
  ofstream ofs(filename, ios::app);
  for(int i = 0; i < size_v; ++i) {
    ofs << v[i] << " ";
  }
  ofs << endl;
  ofs.close();
}

/**
 * Write vector to a file (don't overwrite but append new line).
 */
void write_vector(const char* filename, SWORD* v)
{
  ofstream ofs(filename, ios::app);
  SWORD* w = v;
  while (w->wnum) {
    ofs << w->weight << " ";
    ++w;
  }
  ofs << endl;
  ofs.close();
}

/**
 * Write set of scalar values to a file (don't overwrite but append new lines).
 */
void write_scalars(const char* filename, double* v, int size_v)
{
  ofstream ofs(filename, ios::app);
  for(int i = 0; i < size_v; ++i) {
    ofs << v[i] << endl;
  }
  ofs.close();
}

/**
 * Write scalar value to a file (don't overwrite but append new line).
 */
void write_scalar(const char* filename, double v)
{
  ofstream ofs(filename, ios::app);
  ofs << v << endl;
  ofs.close();
}

/**
 * Returns squared norm
 */
double get_sq_norm(double* v, int _sizePsi)
{
  double sq_norm_v = 0;
  for(int i = 1; i < _sizePsi; ++i) {
    sq_norm_v += v[i]*v[i];
  }
  return sq_norm_v;
}

double get_norm(double* v, int _sizePsi)
{
  double norm_v = 0;
  for(int i = 1; i < _sizePsi; ++i) {
    norm_v += v[i]*v[i];
  }
  return sqrt(norm_v);
}

/**
 * Compute the average norm of psi over the training data
 */
double get_norm_psi_gt(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm, EXAMPLE *examples, long nExamples)
{
  int _sizePsi = sm->sizePsi + 1;
  SWORD* fy_to = new SWORD[_sizePsi];
  double avg_norm = 0;

  for(long i = 0; i < nExamples; ++i) {
    computePsi(fy_to, examples[i].x, examples[i].y, sm, sparm);
    double norm_wy_to = 0;
    SWORD* wy_to = fy_to;
    while (wy_to->wnum) {
      norm_wy_to += wy_to->weight*wy_to->weight;
      ++wy_to;
    }
    norm_wy_to = sqrt(norm_wy_to);
    avg_norm += norm_wy_to;
  }
  avg_norm /= nExamples;

  delete[] fy_to;
  return avg_norm;
}

/**
 * accumulate gradient in dfy
 */
void compute_gradient_accumulate(STRUCTMODEL *sm, GRADIENT_PARM* gparm,
                                 SWORD* fy_to, SWORD* fy_away, double *dfy,
                                 const double loss, const double dfy_weight)
{
#if CUSTOM_VERBOSITY > 2
  double score_y = 0;
  double score_y_away = 0;
#endif

  SWORD* wy_to = fy_to;
  SWORD* wy_away = fy_away;
  switch(gparm->loss_type)
    {
    case LOG_LOSS:
      {
        // L(w) = log(1+e(m(x)))
        // where m(x) = (loss(y,y_bar) + score(x,y_bar)) - score(x,y)
        // and score(x,y) = w^T*psi(x,y)
        // dL(w)/dw = ( m'(x) e(m(x)) ) / ( 1 + e(m(x)))
        // m'(x) = psi(x,y_bar) - psi(x,y)
        double m = 0;
        double dm;
        while (wy_to->wnum) {
          while(wy_away->wnum && (wy_away->wnum < wy_to->wnum)) {
            ++wy_away;
          }

          if(wy_to->wnum == wy_away->wnum) {
            dm = wy_away->weight - wy_to->weight;
          } else {
            dm = - wy_to->weight;
          }
          m += (sm->w[wy_to->wnum]*dm);
          ++wy_to;
        }
        m += loss;
        double e_m = 0;
        if(m < 100) {
          e_m = exp(m);
        }

        wy_to = fy_to;
        wy_away = fy_away;
        while (wy_to->wnum) {
          while(wy_away->wnum && (wy_away->wnum < wy_to->wnum)) {
            ++wy_away;
          }
          if(wy_to->wnum == wy_away->wnum) {
            dm = wy_away->weight - wy_to->weight;
          } else {
            dm = - wy_to->weight;
          }

          if(m >= 100) {
            dfy[wy_to->wnum] += dfy_weight * dm;
          } else {
            dfy[wy_to->wnum] += dfy_weight * (dm*e_m / (e_m + 1));
          }
#if CUSTOM_VERBOSITY > 2
          score_y += sm->w[wy_to->wnum]*wy_to->weight;
          score_y_away += sm->w[wy_to->wnum]*wy_away->weight;
#endif
          ++wy_to;
        }
      }
      break;
    case HINGE_LOSS:
      {
        // L(w) = (loss(y,y_bar) + score(x,y_bar)) - score(x,y)
        // where score(x,y) = w^T*psi(x,y)
        // dL(w)/dw = psi(x,y_bar) - psi(x,y)
        double dm;
        while (wy_to->wnum) {
          while(wy_away->wnum && (wy_away->wnum < wy_to->wnum)) {
            ++wy_away;
          }

          if(wy_to->wnum == wy_away->wnum) {
            dm = wy_away->weight - wy_to->weight;
          } else {
            dm = - wy_to->weight;
          }

          dfy[wy_to->wnum] += dfy_weight * dm;
#if CUSTOM_VERBOSITY > 2
          score_y += sm->w[wy_to->wnum]*wy_to->weight;
          score_y_away += sm->w[wy_to->wnum]*wy_away->weight;
#endif
          ++wy_to;
        }
      }
      break;
    case SQUARE_HINGE_LOSS:
      {
        // L(w) = log(1+e(m(x)))
        // where m(x) = (loss(y,y_bar) + score(x,y_bar)) - score(x,y)
        // and score(x,y) = w^T*psi(x,y)
        // dL(w)/dw = ( m'(x) e(m(x)) ) / ( 1 + e(m(x)))
        // m'(x) = psi(x,y_bar) - psi(x,y)
        double m = 0;
        double dm;
        while (wy_to->wnum) {
          while(wy_away->wnum && (wy_away->wnum < wy_to->wnum)) {
            ++wy_away;
          }

          if(wy_to->wnum == wy_away->wnum) {
            dm = wy_away->weight - wy_to->weight;
          } else {
            dm = - wy_to->weight;
          }
          m += (sm->w[wy_to->wnum]*dm);
          ++wy_to;
        }
        m += loss;

        wy_to = fy_to;
        wy_away = fy_away;
        while (wy_to->wnum) {
          while(wy_away->wnum && (wy_away->wnum < wy_to->wnum)) {
            ++wy_away;
          }
          if(wy_to->wnum == wy_away->wnum) {
            dm = wy_away->weight - wy_to->weight;
          } else {
            dm = - wy_to->weight;
          }

          dfy[wy_to->wnum] += 1e-30 * dfy_weight * dm * m;
#if CUSTOM_VERBOSITY > 2
          score_y += sm->w[wy_to->wnum]*wy_to->weight;
          score_y_away += sm->w[wy_to->wnum]*wy_away->weight;
#endif
          ++wy_to;
        }
      }
      break;
    default:
      printf("[svm_struct_custom] Unknown loss type %d\n", gparm->loss_type);
      exit(-1);
      break;
    }

#if CUSTOM_VERBOSITY > 2
  ofstream ofs_score_y("score_y.txt", ios::app);
  ofs_score_y << score_y << endl;
  ofs_score_y.close();

  ofstream ofs_score_y_away("score_y_away.txt", ios::app);
  ofs_score_y_away << score_y_away << endl;
  ofs_score_y_away.close();
#endif
}

double computeScore(STRUCTMODEL *sm, SWORD* fy)
{
  double score = 0;
  double* smw = sm->w + 1;
  SWORD* w = fy;
  while (w->wnum) {
    score += w->weight*smw[w->wnum];
    ++w;
  }
  return score;
}


void compute_psi(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                   EXAMPLE* ex, LABEL* y_bar, LABEL* y_direct,
                   GRADIENT_PARM* gparm, SWORD* fy_to, SWORD* fy_away,
                   double* loss)
{
  labelType* y_to = 0;
  labelType* y_away = 0;
  switch(gparm->gradient_type) {
  case GRADIENT_GT:
    // moves toward ground truth, away from larger loss
    y_to = ex->y.nodeLabels;
    y_away = y_bar->nodeLabels;
    computePsi(fy_to, ex->x, ex->y, sm, sparm);
    computePsi(fy_away, ex->x, *y_bar, sm, sparm);
    break;
  case GRADIENT_DIRECT_ADD:
    // moves away from larger loss
    y_to = y_direct->nodeLabels;
    y_away = y_bar->nodeLabels;
    computePsi(fy_to, ex->x, *y_direct, sm, sparm);
    computePsi(fy_away, ex->x, *y_bar, sm, sparm);
    break;
  case GRADIENT_DIRECT_SUBTRACT:
    // moves toward better label
    y_to = y_direct->nodeLabels;
    y_away = y_bar->nodeLabels;
    computePsi(fy_to, ex->x, *y_direct, sm, sparm);
    computePsi(fy_away, ex->x, *y_bar, sm, sparm);
    break;
  default:
    printf("[svm_struct_custom] Unknown gradient type\n");
    exit(-1);
    break;
  }

  if(!gparm->ignore_loss) {
    int nDiff;
    double _loss;
    computeLoss(y_to, y_away, ex->y.nNodes, sparm, _loss, nDiff);
    if(loss) {
      *loss = _loss;
    }
  }
}

void compute_psi_to(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                    EXAMPLE* ex, GRADIENT_PARM* gparm, SWORD* fy_to)
{
  switch(gparm->gradient_type) {
  case GRADIENT_GT:
    // moves toward ground truth, away from larger loss
    computePsi(fy_to, ex->x, ex->y, sm, sparm);
    break;
    /*
  case GRADIENT_DIRECT_ADD:
    // moves away from larger loss
    computePsi(fy_to, ex->x, *y_direct, sm, sparm);
    break;
  case GRADIENT_DIRECT_SUBTRACT:
    // moves toward better label
    computePsi(fy_to, ex->x, *y_direct, sm, sparm);
    break;
    */
  default:
    printf("[svm_struct_custom] Unknown gradient type\n");
    exit(-1);
    break;
  }
}

double compute_gradient_accumulate(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                                   EXAMPLE* ex, LABEL* y_bar, LABEL* y_direct,
                                   GRADIENT_PARM* gparm, SWORD* fy_to, SWORD* fy_away,
                                   double *dfy, double* loss, const double dfy_weight)
{
  int _sizePsi = sm->sizePsi + 1;
  double _loss;
  compute_psi(sparm, sm, ex, y_bar, y_direct, gparm, fy_to, fy_away, &_loss);
  if(loss) {
    *loss = _loss;
  }

  compute_gradient_accumulate(sm, gparm, fy_to, fy_away, dfy, _loss, dfy_weight);

#if CUSTOM_VERBOSITY > 3
  write_vector("dfy.txt", dfy, _sizePsi);
#endif

  double dscore = 0;
  // do not add +1 here as dfy also has an additional dummy entry at index 0.
  double* smw = sm->w;
  for(int i = 0; i < _sizePsi; ++i) {
    dscore += smw[i]*dfy[i];
  }
  return dscore;
}

double compute_gradient(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                        EXAMPLE* ex, LABEL* y_bar, LABEL* y_direct,
                        GRADIENT_PARM* gparm, SWORD* fy_to, SWORD* fy_away,
                        double *dfy, double* loss, const double dfy_weight)
{
  // initialize dfy to 0
  int _sizePsi = sm->sizePsi + 1;
  for(int i = 0; i < _sizePsi; ++i) {
    dfy[i] = 0;
  }

  return compute_gradient_accumulate(sparm, sm, ex, y_bar, y_direct, gparm, fy_to, fy_away, dfy, loss, dfy_weight);
}

double compute_gradient(STRUCTMODEL *sm, GRADIENT_PARM* gparm,
                        SWORD* fy_to, SWORD* fy_away, double *dfy,
                        const double loss, const double dfy_weight)
{
  // initialize dfy to 0
  int _sizePsi = sm->sizePsi + 1;
  for(int i = 0; i < _sizePsi; ++i) {
    dfy[i] = 0;
  }

  compute_gradient_accumulate(sm, gparm, fy_to, fy_away, dfy, loss, dfy_weight);

  double dscore = 0;
  // do not add +1 here as dfy also has an additional dummy entry at index 0.
  double* smw = sm->w;
  for(int i = 0; i < _sizePsi; ++i) {
    dscore += smw[i]*dfy[i];
  }
  return dscore;
}

void exportLabels(STRUCT_LEARN_PARM *sparm, EXAMPLE* ex,
                  LABEL* y, const char* dir_name)
{
  string paramSlice3d;
  Config::Instance()->getParameter("slice3d", paramSlice3d);
  bool useSlice3d = paramSlice3d.c_str()[0] == '1';
  string paramVOC;
  Config::Instance()->getParameter("voc", paramVOC);
  bool useVOC = paramVOC.c_str()[0] == '1';

  stringstream ss_dir;
  ss_dir << dir_name;
  mkdir(ss_dir.str().c_str(), 0777);
  if(!useSlice3d) {
    //TODO : Remove !useVOC
    if(useVOC) {
      ss_dir << "x" << sparm->iterationId;
    }
    else {
      ss_dir << "x" << sparm->iterationId << "/";
    }
  }
  mkdir(ss_dir.str().c_str(), 0777);

  stringstream soutColoredImage;
  soutColoredImage << ss_dir.str();
  if(useSlice3d) {
    soutColoredImage << getNameFromPathWithoutExtension(ex->x.slice->getName());
    soutColoredImage << "_";
    soutColoredImage << sparm->iterationId;
  } else {
    soutColoredImage << ex->x.slice->getName();
  }

  ex->x.slice->exportSupernodeLabels(soutColoredImage.str().c_str(),
                                     sparm->nClasses,
                                     y->nodeLabels,
                                     y->nNodes,
                                     &(sparm->labelToClassIdx));

  if(useSlice3d) {
    zipAndDeleteCube(soutColoredImage.str().c_str());
  }
}

double do_gradient_step(STRUCT_LEARN_PARM *sparm,
                        STRUCTMODEL *sm, EXAMPLE *ex, long nExamples,
                        GRADIENT_PARM* gparm,
                        double* momentum, double& dscore, LABEL* y_bar)
{
  int _sizePsi = sm->sizePsi + 1;
  SWORD* fy_to = new SWORD[_sizePsi];
  SWORD* fy_away = new SWORD[_sizePsi];
  double* dfy = new double[_sizePsi];
  memset((void*)dfy, 0, sizeof(double)*(_sizePsi));

  double m = do_gradient_step(sparm, sm, ex, nExamples, gparm,
                              momentum, fy_to, fy_away, dfy, dscore, y_bar);
  delete[] fy_to;
  delete[] fy_away;
  delete[] dfy;
  return m;
}

double compute_gradient_with_history(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                                     EXAMPLE* ex,
                                     GRADIENT_PARM* gparm, SWORD* fy_to,
                                     double *dfy, double* loss)
{
  ConstraintSet* cs = ConstraintSet::Instance();
  const vector< constraint >* constraints = cs->getConstraints(ex->x.id);
  assert(constraints != 0);
  int n_cs = constraints->size();

  double* dfy_weights = new double[n_cs];
  if(gparm->use_random_weights) {
    double total_weights = 0;
    for(int c = 0; c < n_cs; ++c) {
      dfy_weights[c] = rand() * ((double)n_cs/(double)RAND_MAX);
      total_weights += dfy_weights[c];
    }
    for(int c = 0; c < n_cs; ++c) {
      dfy_weights[c] /= total_weights;
    }

    printf("dfy_weights:\n");
    for(int c = 0; c < n_cs; ++c) {
      printf("%g ", dfy_weights[c]);
    }
    printf("\n");

  } else {
    double total_weights = 0;
    for(int c = 0; c < n_cs; ++c) {
      dfy_weights[c] = 1.0/(double)(n_cs);
      total_weights += dfy_weights[c];
    }
    for(int c = 0; c < n_cs; ++c) {
      dfy_weights[c] /= total_weights;
    }
  }

  int _sizePsi = sm->sizePsi + 1;
  // initialize dfy to 0
  for(int i = 0; i < _sizePsi; ++i) {
    dfy[i] = 0;
  }

  if(loss) {
    *loss = 0;
  }

  // add gradient for history of constraints
  if(gparm->loss_type != HINGE_LOSS && gparm->loss_type != SQUARE_HINGE_LOSS) {
    // use all the constraints in the set
    int c = 0;
    for(vector<constraint>::const_iterator it = constraints->begin();
        it != constraints->end(); ++it) {
      compute_gradient_accumulate(sm, gparm, fy_to,
                                  it->first->w, dfy, it->first->loss, dfy_weights[c]);
      if(loss) {
        *loss += it->first->loss;
      }      
      ++c;
    }
  } else {
    // only use violated constraints

    double score_gt = computeScore(sm, fy_to);
    int c = 0;
    for(vector<constraint>::const_iterator it = constraints->begin();
        it != constraints->end(); ++it) {
      // check if constraint is violated
      double score_cs = computeScore(sm, it->first->w);
      bool positive_margin = (score_cs - score_gt + it->first->loss) > 0;
      //printf("Margin constraint %d: score_cs = %g, score_gt = %g, loss = %g, margin = %g\n",
      //       c, score_cs, score_gt, it->first->loss, score_cs - score_gt + it->first->loss);

      if(positive_margin) {
        compute_gradient_accumulate(sm, gparm, fy_to,
                                    it->first->w, dfy, it->first->loss, dfy_weights[c]);
        if(loss) {
          *loss += it->first->loss;
        }
      }     
      ++c;
    }
  }

  double total_dscore = 0;
  // do not add +1 here as dfy also has an additional dummy entry at index 0.
  double* smw = sm->w;
  for(int i = 0; i < _sizePsi; ++i) {
    total_dscore += smw[i]*dfy[i];
  }

  delete[] dfy_weights;

  return total_dscore;
}

double compute_gradient_with_history(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                                     EXAMPLE* ex, LABEL* y_bar, LABEL* y_direct,
                                     GRADIENT_PARM* gparm, SWORD* fy_to, SWORD* fy_away,
                                     double *dfy, double* loss)
{
  double dfy_weight = 1.0;
  int _sizePsi = sm->sizePsi + 1;
  // initialize dfy to 0
  for(int i = 0; i < _sizePsi; ++i) {
    dfy[i] = 0;
  }

  double _loss;
  double _dscore = compute_gradient(sparm, sm, ex, y_bar, y_direct, gparm, fy_to,
                                    fy_away, dfy, &_loss, dfy_weight);
  if(loss) {
    *loss += _loss;
  }

  // add gradient for history of constraints
  ConstraintSet* cs = ConstraintSet::Instance();
  const vector< constraint >* constraints = cs->getConstraints(ex->x.id);
  if(constraints) {
    dfy_weight = 1.0/(double)(constraints->size()+1.0);
    for(vector<constraint>::const_iterator it = constraints->begin();
        it != constraints->end(); ++it) {
      compute_gradient_accumulate(sm, gparm, fy_to,
                                  it->first->w, dfy, it->first->loss, dfy_weight);
      if(loss) {
        *loss += it->first->loss;
      }
    }
  }

  double total_dscore = 0;
  // do not add +1 here as dfy also has an additional dummy entry at index 0.
  double* smw = sm->w;
  for(int i = 0; i < _sizePsi; ++i) {
    total_dscore += smw[i]*dfy[i];
  }
  total_dscore += _dscore;
  return total_dscore;
}

/* Compute the upper envelope of a set of n (non-vertical) lines, y = a*x + b.
   - Input: n a's and b's
   - Output: at most n-1 x's and y's for the intersections, sorted by x, and the min and max a's if needed
 */
vector<pair<double,double> > computeLineUpperEnvelope(const vector<pair<double,double> > &lines_ab, double *a_min, double *a_max, vector<int>& line_indices)
{
  typedef pair<double,double> Pair;
  
  assert(!lines_ab.empty());

  int n = lines_ab.size();
  vector< pair<Pair, int> > sorted_ab(n);
  for(int i = 0; i < n; ++i) {
    sorted_ab[i] = make_pair(lines_ab[i], i);
  }

  // Negate b and later negate it back, so b is in decreasing order if a is tied after sorting
  for(int i = 0; i<n; i++) {
    sorted_ab[i].first.second = -sorted_ab[i].first.second;
  }

  sort(sorted_ab.begin(), sorted_ab.end());  // sorted according to a and then b (if a tied)

  for(int i=0; i<n; i++) {
    sorted_ab[i].first.second = -sorted_ab[i].first.second;
  }

  // Record min and max a's
  if(a_min) {
    *a_min = sorted_ab.front().first.first;
  }
  if(a_max) {
    *a_max = sorted_ab.back().first.first;
  }

  vector<int> rel_lines_idx;  // relevant lines (index in the sorted_ab array)
  vector<Pair> rel_intersections;  // relevant intersections

  rel_lines_idx.push_back(0);  // first line is always relevant

  // Main loop
  for(int i=1; i<n; i++) {
    Pair curr_ab = sorted_ab[i].first;
    Pair prev_ab = sorted_ab[rel_lines_idx.back()].first;
    double a1 = curr_ab.first, a2 = prev_ab.first;
    double b1 = curr_ab.second, b2 = prev_ab.second;

    if(a1 == a2) {
      continue;  // current line is irrelevant, since current b <= previous b (guaranteed by sorting)
    }

    // Line intersection between y = a1 * x + b1 and y = a1 * x + b2
    double x = (b2 - b1) / (a1 - a2);
    double y = (a1 * b2 - a2 * b1) / (a1 - a2);

    // If the new intersection is to the left of the previous intersection,
    // the previous intersection and the previous line become irrelevant
    // for the upper envelope.  Remove them and recurse.
    while(!rel_intersections.empty() && x <= rel_intersections.back().first) {
      rel_intersections.pop_back();
      rel_lines_idx.pop_back();

      prev_ab = sorted_ab[rel_lines_idx.back()].first;
      a2 = prev_ab.first;
      b2 = prev_ab.second;
      x = (b2 - b1) / (a1 - a2);
      y = (a1 * b2 - a2 * b1) / (a1 - a2);
    }

    rel_intersections.push_back(Pair(x, y));
    rel_lines_idx.push_back(i);
  }

  for(uint i = 0; i < rel_lines_idx.size(); ++i) {
    line_indices.push_back(sorted_ab[rel_lines_idx[i]].second);
  }

  return rel_intersections;
}

/* Compute the sum of two series of connected line segments, which is itself a series of connected
   line segments.
   - A series of N line segments are parameterized by a vector of N pairs.  The first N-1 pairs are
   the (x, y) coordinates of the intersection (i.e. connecting) points, while the last pair
   is (a_min, a_max), i.e. the slopes of the first and last line segments (which by definition
   have the smallest and largest slopes, respectively).
   - Assume all intersection points are *SORTED* by their x-coordinates.
*/
vector<pair<double,double> > addSeriesOfLineSegments(const vector<pair<double,double> > &series1, const vector<pair<double,double> > &series2)
{
  if(!series1.size())
    return series2;
  if(!series2.size())
    return series1;

  int n1 = series1.size() - 1, n2 = series2.size() - 1;

  // Find for each intersection on one series the corresponding (to same x location) value on the other series
  vector<double> shift1(n1), shift2(n2);

  const pair<double,double> *series[2] = {&series1[0], &series2[0]};
  double *shift[2] = {&shift1[0], &shift2[0]};
  int n[2] = {n1, n2};

  for(int h=0; h<2; h++) {
    int other_ind = 0;
    for(int i=0; i<n[h]; i++) {
      while(other_ind < n[1-h] && series[1-h][other_ind].first <= series[h][i].first) {
        other_ind++;
      }
      if(other_ind == 0) {
        shift[h][i] = series[1-h][0].second + series[1-h][n[1-h]].first * (series[h][i].first - series[1-h][0].first);
      }
      else if(other_ind == n[1-h]) {
        shift[h][i] = series[1-h][n[1-h] - 1].second + series[1-h][n[1-h]].second * (series[h][i].first - series[1-h][n[1-h] - 1].first);
      }
      else {
        shift[h][i] = (series[1-h][other_ind - 1].second * (series[1-h][other_ind].first - series[h][i].first) +
                       series[1-h][other_ind].second * (series[h][i].first - series[1-h][other_ind - 1].first)) / (series[1-h][other_ind].first - series[1-h][other_ind - 1].first);
      }
    }
  }

  // The new set of intersections are from either series1 or series2.
  vector<pair<double,double> > result(n1 + n2 + 1);
  int i1 = 0, i2 = 0;
  for(int i=0; i<n1 + n2; i++) {
    if(i2 < n2 && (i1 == n1 || series2[i2].first < series1[i1].first)) {
      result[i].first = series2[i2].first;
      result[i].second = series2[i2].second + shift2[i2];
      i2++;
    }
    else {
      assert(i1 < n1);
      result[i].first = series1[i1].first;
      result[i].second = series1[i1].second + shift1[i1];
      i1++;
    }
  }
  assert(i1 == n1);
  assert(i2 == n2);

  result[n1 + n2].first = series1[n1].first + series2[n2].first;
  result[n1 + n2].second = series1[n1].second + series2[n2].second;

  return result;
}

void computeUpperEnveloppeForOneExample(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                                        GRADIENT_PARM* gparm,
                                        EXAMPLE* examples, int il, SWORD* fy_to,
                                        double *dfy, vector<pair<double,double> >& lines_ab,
                                        vector<int>& line_indices,
                                        vector<pair<double,double> >& rel_intersections,
                                        double* a_min, double* a_max) {
  ConstraintSet* cs = ConstraintSet::Instance();
  const vector< constraint >* constraints = cs->getConstraints(examples[il].x.id);

  if(!constraints) {
    return;
  }

  int _sizePsi = sm->sizePsi + 1;
  // psi_groundtruth - psi_constraint
  double* delta_c = new double[_sizePsi];

  for(vector<constraint>::const_iterator it = constraints->begin();
      it != constraints->end(); ++it) {

    for(int i = 0; i < _sizePsi; ++i) {
      delta_c[i] = 0;
    }

    SWORD* wy_to = fy_to;
    SWORD* wy_away = it->first->w;
    while (wy_to->wnum) {
      while(wy_away->wnum && (wy_away->wnum < wy_to->wnum)) {
        ++wy_away;
      }

      // delta_c = psi_constraint - psi_groundtruth
      // wy_to = psi_groundtruth
      // wy_away = psi_constraint
      delta_c[wy_to->wnum] = wy_away->weight - wy_to->weight;
      ++wy_to;
    }

    // project constraint delta_c on gradient direction dfy
    double a = 0;
    for(int i = 0; i < _sizePsi; ++i) {
      a += -dfy[i]*delta_c[i];
    }

    // project current w on constraint dfy_c
    double b = 0;
    for(int i = 0; i < _sizePsi; ++i) {
      b += sm->w[i]*delta_c[i];
    }
    b += it->first->loss;

    if(gparm->autostep_regularization_type == REGULARIZATION_SVM) {

      double a_svm = 0;
      for(int i = 0; i < _sizePsi; ++i) {
        a_svm += -dfy[i]*sm->w[i];
      }
      a += 2*gparm->autostep_regularization*a_svm;

      b += gparm->autostep_regularization*get_sq_norm(sm->w, _sizePsi);
    }

    lines_ab.push_back(make_pair(a,b));
  }

  // add (0,0) constraint
  lines_ab.push_back(make_pair(0,0));

  rel_intersections = computeLineUpperEnvelope(lines_ab, a_min, a_max, line_indices);

  delete[] delta_c;
}

void computeUpperEnveloppe_max(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                               GRADIENT_PARM* gparm,
                               double *dfy, vector<pair<double,double> >& lines_ab,
                               vector<int>& line_indices,
                               vector<pair<double,double> >& rel_intersections,
                               double* a_min, double* a_max) {
  ConstraintSet* cs = ConstraintSet::Instance();
  int _sizePsi = sm->sizePsi + 1;
  // psi_groundtruth - psi_constraint
  double* delta_c = new double[_sizePsi];
  SWORD* fy_to = new SWORD[_sizePsi];

  EXAMPLE* examples = gparm->examples_all;
  long nExamples = gparm->n_total_examples;
  for(int il = 0; il < nExamples; ++il) {

    const vector< constraint >* constraints = cs->getConstraints(examples[il].x.id);
    if(constraints == 0) {
      continue;
    }

    compute_psi_to(sparm, sm, &examples[il], gparm, fy_to);

    for(vector<constraint>::const_iterator it = constraints->begin();
        it != constraints->end(); ++it) {

      for(int i = 0; i < _sizePsi; ++i) {
        delta_c[i] = 0;
      }

      SWORD* wy_to = fy_to;
      SWORD* wy_away = it->first->w;
      while (wy_to->wnum) {
        while(wy_away->wnum && (wy_away->wnum < wy_to->wnum)) {
          ++wy_away;
        }

        // delta_c = psi_constraint - psi_groundtruth
        // wy_to = psi_groundtruth
        // wy_away = psi_constraint
        delta_c[wy_to->wnum] = wy_away->weight - wy_to->weight;
        ++wy_to;
      }

      // project constraint delta_c on gradient direction dfy
      double a = 0;
      for(int i = 0; i < _sizePsi; ++i) {
        a += -dfy[i]*delta_c[i];
      }

      // project current w on constraint dfy_c
      double b = 0;
      for(int i = 0; i < _sizePsi; ++i) {
        b += sm->w[i]*delta_c[i];
      }
      b += it->first->loss;

      if(gparm->autostep_regularization_type == REGULARIZATION_SVM) {

        double a_svm = 0;
        for(int i = 0; i < _sizePsi; ++i) {
          a_svm += -dfy[i]*sm->w[i];
        }
        a += 2*gparm->autostep_regularization*a_svm;

        b += gparm->autostep_regularization*get_sq_norm(sm->w, _sizePsi);
      }

      lines_ab.push_back(make_pair(a,b));
    }
  }

  // add (0,0) constraint
  lines_ab.push_back(make_pair(0,0));

  rel_intersections = computeLineUpperEnvelope(lines_ab, a_min, a_max, line_indices);

  delete[] delta_c;
  delete[] fy_to;
}

// return number of processed examples
int computeUpperEnveloppe_sum(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                              GRADIENT_PARM* gparm, double *dfy,
                              vector<pair<double,double> >& lines_ab,
                              vector<int>& line_indices,
                              vector<pair<double,double> >& series_sum) {
  int nProcessedExamples = 0;
  EXAMPLE* examples = gparm->examples_all;
  long nExamples = gparm->n_total_examples;

  series_sum.clear();

  SWORD* fy_to = new SWORD[sm->sizePsi+1];

  // compute sum for all examples
  for(int il = 0; il < nExamples; ++il) {

    vector<int> _line_indices;
    vector<pair<double,double> > _lines_ab;

    compute_psi_to(sparm, sm, &examples[il], gparm, fy_to);

    vector<pair<double,double> > series_example_i;
    double a_min = 0;
    double a_max = 0;
    computeUpperEnveloppeForOneExample(sparm, sm, gparm, examples, il, fy_to,
                                       dfy, _lines_ab, _line_indices,
                                       series_example_i, &a_min, &a_max);

    //-------------------
    // debug
    for(int i = 0; i < ((long)_line_indices.size()) - 1; ++i) {
      if( (_lines_ab[_line_indices[i]].first - _lines_ab[_line_indices[i+1]].first) > 1e-8) {
        printf("WARNING: _lines_ab[_line_indices[i]].first - _lines_ab[_line_indices[i+1]].first > 0 %g %g\n",
               _lines_ab[_line_indices[i]].first, _lines_ab[_line_indices[i+1]].first);
      }
    }

    /*
    printf("series_example_i %ld\n", series_example_i.size());
    for(vector<pair<double,double> >::iterator it = series_example_i.begin(); it != series_example_i.end(); ++it) {
      printf("(%g, %g) ", it->first, it->second);
    }
    printf("\n");
    */
    //-------------------

    if(series_example_i.size() > 0) {

      ++nProcessedExamples;

      // debug      
      if(_line_indices.size() > _lines_ab.size()) {
        printf("_line_indices.size() > _lines_ab.size() %ld %ld\n", _line_indices.size(), _lines_ab.size());
        exit(-1);
      }

      // sanity check
      double _a_min = _lines_ab[_line_indices[0]].first;
      double _a_max = _lines_ab[_line_indices[_line_indices.size()-1]].first;
      assert(_a_min==a_min);
      assert(_a_max==a_max);

      // add (a_min, a_max), i.e. the slopes of the first and last line segments
      series_example_i.push_back(make_pair(a_min, a_max));

      series_sum = addSeriesOfLineSegments(series_sum, series_example_i);
    } else {
      printf("[svm_struct_custom] WARNING: series_example_i.size() = 0 at iteration %d\n",
             sparm->iterationId);
    }

  }

  delete[] fy_to;

  // add a_min
  double a_min = series_sum[series_sum.size() - 1].first;
  double b_min = series_sum[0].second - a_min*series_sum[0].first; // y - ax
  lines_ab.push_back(make_pair(a_min,b_min));
  line_indices.push_back(0);

  // iterate over all segments and compute a's and b's (s.t. y = ax+b).
  // don't use the last pair that contains (a_min, a_max)
  for(long i = 0; i < ((long)series_sum.size()) - 2; ++i) {
    double x1 = series_sum[i].first;
    double y1 = series_sum[i].second;
    double x2 = series_sum[i+1].first;
    double y2 = series_sum[i+1].second;

    //printf("(x1,y1) = (%g,%g), (x2,y2) = (%g,%g)\n", x1, y1, x2, y2);

    double dx = x2 - x1;

    double a = 0;
    if(dx != 0) {
      a = (y2 - y1)/dx;
    }

    // debug
    if(x1 > x2) {
      printf("WARNING: x1 > x2 %g %g\n", x1, x2);
    }

    double b = y2 - a*x2;

    lines_ab.push_back(make_pair(a,b));
    line_indices.push_back(i+1);
  }

  // add a_max
  double a_max = series_sum[series_sum.size() - 1].second;
  double b_max = series_sum[series_sum.size() - 2].second - a_max*series_sum[series_sum.size() - 2].first;  // y - ax
  lines_ab.push_back(make_pair(a_max,b_max));
  line_indices.push_back(series_sum.size()-1);

  // remove last pair (a_min, a_max)
  series_sum.resize(series_sum.size() - 1);

  return nProcessedExamples;
}

double compute_optimal_step_size(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm, GRADIENT_PARM* gparm,
                                 EXAMPLE* examples, int il, SWORD* fy_to, double *dfy)
{
  double step_size = -1;

  vector<int> line_indices;
  vector<pair<double,double> > lines_ab;
  vector<pair<double,double> > rel_intersections;
  int nProcessedExamples = 1;

  if(gparm->autostep_obj_type == OBJECTIVE_ONE_EXAMPLE ||
     gparm->n_total_examples == 1 || sparm->iterationId <= 1) {
    computeUpperEnveloppeForOneExample(sparm, sm, gparm, examples, il, fy_to,
                                       dfy, lines_ab, line_indices, rel_intersections, 0, 0);
  } else {
    if(gparm->autostep_obj_type == OBJECTIVE_SUM) {
      nProcessedExamples = computeUpperEnveloppe_sum(sparm, sm, gparm, dfy, lines_ab, line_indices, rel_intersections);
    } else {
      computeUpperEnveloppe_max(sparm, sm, gparm, dfy, lines_ab, line_indices, rel_intersections, 0, 0);
    }
  }

  //-----------------
  // debug
  for(int i = 0; i < ((long)line_indices.size()) - 1; ++i) {
    if( (lines_ab[line_indices[i]].first - lines_ab[line_indices[i+1]].first) > 1e-8) {
      printf("WARNING: %d lines_ab[line_indices[i]].first - lines_ab[line_indices[i+1]].first > 0 %g %g\n",
             i, lines_ab[line_indices[i]].first, lines_ab[line_indices[i+1]].first);
    }
  }

  /*
  printf("rel_intersections %ld\n", rel_intersections.size());
  for(vector<pair<double,double> >::iterator it = rel_intersections.begin(); it != rel_intersections.end(); ++it) {
    printf("(%g, %g) ", it->first, it->second);
  }
  printf("\n");
  */
  //----------------

  switch(gparm->autostep_regularization_type) {
  case REGULARIZATION_NONE:
    {
      double min_y = DBL_MAX;
      double matching_x = -1;
      for(vector<pair<double,double> >::iterator it = rel_intersections.begin();
          it != rel_intersections.end(); ++it) {
        if(it->second < min_y) {
          // if y is equal to the min y found so far,
          // pick smallest x
          //if( (it->second < min_y) || (matching_x > it->first)) {
          min_y = it->second;
          matching_x = it->first;
        }
      }

      /*
      // --------------------
      // sanity check for linear segments
      // make sure min_y_1 = min_y_2 at minimum
      // todo: compute min_idx in loop above
      {
        int line_idx_1 = line_indices[min_idx];
        int line_idx_2 = line_indices[min_idx + 1];

        double min_x_1 = matching_x;
        double min_x_2 = matching_x;
        double min_y_1 = lines_ab[line_idx_1].first*min_x_1 + lines_ab[line_idx_1].second;
        double min_y_2 = lines_ab[line_idx_2].first*min_x_2 + lines_ab[line_idx_2].second;

        printf("[svm_struct_custom] linear idx %d: %g*x + %g -> (%g, %g)\n", line_idx_1, lines_ab[line_idx_1].first, lines_ab[line_idx_1].second, min_x_1, min_y_1);
        printf("[svm_struct_custom] linear idx %d: %g*x + %g -> (%g, %g)\n", line_idx_2, lines_ab[line_idx_2].first, lines_ab[line_idx_2].second, min_x_2, min_y_2);
      }
      // --------------------
      */

      step_size = matching_x;
      printf("[svm_struct_custom] autostep linear min (%g,%g)\n", matching_x, min_y);

#if CUSTOM_VERBOSITY > 1
      ofstream ofs_autostep_min("autostep_linear_min.txt", ios::app);
      ofs_autostep_min << matching_x << " " << min_y << endl;
      ofs_autostep_min.close();
#endif
    }
    break;
  case REGULARIZATION_PA:
  case REGULARIZATION_PA_TIME_ADAPTIVE:
  case REGULARIZATION_SVM:
    {
      int _sizePsi = sm->sizePsi + 1;
      double sq_norm_dfy = get_sq_norm(dfy, _sizePsi);
      // go through all the intersections and add regularization
      vector< pair<double, int> > y_values;
      double quadratic_coeff = nProcessedExamples*gparm->autostep_regularization*sq_norm_dfy;
      if(gparm->autostep_regularization_type == REGULARIZATION_PA_TIME_ADAPTIVE) {
        quadratic_coeff *= pow(sparm->iterationId+1, gparm->autostep_regularization_exp);
      }

      int idx = 0;
      for(vector<pair<double,double> >::iterator it = rel_intersections.begin();
          it != rel_intersections.end(); ++it) {
        double x = it->first;
        double y = it->second + quadratic_coeff*x*x;
        y_values.push_back(make_pair(y, idx));
        ++idx;
      }

#if 0
      vector< pair<double, int> >::iterator it_min = std::min(y_values.begin(), y_values.end());
      int min_idx = it_min->second;
      double min_int_y = it_min->first;
#else
      int min_idx =  y_values[0].second;
      double min_int_y = y_values[0].first; 
      for(vector< pair<double, int> >::iterator it = y_values.begin(); it != y_values.end(); ++it) {
        if(min_int_y > it->first) {
          min_int_y = it->first;
          min_idx = it->second;
        }
      }
#endif

      // x corresponding to the lowest intersection point of the quadratic segment upper envelope
      double q_int_x = rel_intersections[min_idx].first;
      int line_idx_1 = line_indices[min_idx];
      int line_idx_2 = line_indices[min_idx + 1];

      /*
      //YL: TODO: Check the y-values against the line segments (lines_ab) equations
      { //DBBUG
        printf("it->second: %e, from line equation: %e, %e\n", min_int_y-quadratic_coeff*q_int_x*q_int_x,
               lines_ab[line_idx_1].first * q_int_x + lines_ab[line_idx_1].second,
               lines_ab[line_idx_2].first * q_int_x + lines_ab[line_idx_2].second);
        printf("quadratic term: %e, q_int_x: %e, quadratic_coeff: %e\n", quadratic_coeff*q_int_x*q_int_x, q_int_x, quadratic_coeff);
      }
      */

      //printf("[svm_struct_custom] autostep linear min + shift %d -> (%g,%g)\n", it_min->second, rel_intersections[it_min->second].first, it_min->first);
 
      // --------------------     
      // sanity check for quadratic segments
      // make sure min_y_1 = min_y_2 at minimum
      // note that the linear segments might not be cross at q_int_x anymore since we added the quadratic term.
      {

        double min_x_1 = q_int_x;
        double min_x_2 = q_int_x;
        double min_y_1 = quadratic_coeff*min_x_1*min_x_1 + lines_ab[line_idx_1].first*min_x_1 + lines_ab[line_idx_1].second;
        double min_y_2 = quadratic_coeff*min_x_2*min_x_2 + lines_ab[line_idx_2].first*min_x_2 + lines_ab[line_idx_2].second;
        double df_min_y = min_y_2 - min_y_1;

        if(abs(df_min_y) > 1) {
          printf("[svm_struct_custom] quad idx %d: %g*x^2 + %g*x + %g -> (%g + %g + %g, %g) = (%g, %g)\n",
                 line_idx_1, quadratic_coeff, lines_ab[line_idx_1].first, lines_ab[line_idx_1].second,
                 min_x_1, quadratic_coeff*min_x_1*min_x_1, lines_ab[line_idx_1].first*min_x_1, lines_ab[line_idx_1].second, min_x_1, min_y_1);
          printf("[svm_struct_custom] quad idx %d: %g*x^2 + %g*x + %g -> (%g + %g + %g, %g) = (%g, %g)\n",
                 line_idx_2, quadratic_coeff, lines_ab[line_idx_2].first, lines_ab[line_idx_2].second,
                 min_x_2, quadratic_coeff*min_x_2*min_x_2, lines_ab[line_idx_2].first*min_x_2, lines_ab[line_idx_2].second, min_x_2, min_y_2);
          printf("[svm_struct_custom] quad diff_y = %g\n", df_min_y);
        }
      }
      // --------------------
      

      double min_x_1 = -lines_ab[line_idx_1].first/(2*quadratic_coeff);
      double min_x_2 = -lines_ab[line_idx_2].first/(2*quadratic_coeff);

      double min_y_1 = quadratic_coeff*min_x_1*min_x_1 + lines_ab[line_idx_1].first*min_x_1 + lines_ab[line_idx_1].second;
      double min_y_2 = quadratic_coeff*min_x_2*min_x_2 + lines_ab[line_idx_2].first*min_x_2 + lines_ab[line_idx_2].second;

      bool is_left = false;

      // check if x is on the left side of the first segment
      double min_y = min_int_y;
      if(min_x_1 < q_int_x) {
        step_size = min_x_1;
        min_y = min_y_1;
        is_left = true;
      } else {
        step_size = q_int_x;
      }

      // check if x is on the right side of the second segment
      if(min_x_2 > q_int_x) {
        step_size = min_x_2;
        min_y = min_y_2;
        if(is_left) { printf("[svm_struct_custom] Warning: min_x_1 = %g, min_x_2 = %g, q_int_x = %g", min_x_1, min_x_2, q_int_x); }
      }

      /*
      // -------------------------
      // debug
      printf("[svm_struct_custom] intersection point (%g, %g)\n", q_int_x, min_int_y);
      printf("[svm_struct_custom] quad idx %d: %g*x^2 + %g*x + %g -> (%g, %g + %g + %g) = (%g, %g)\n",
             line_idx_1, quadratic_coeff, lines_ab[line_idx_1].first, lines_ab[line_idx_1].second,
             min_x_1, quadratic_coeff*min_x_1*min_x_1, lines_ab[line_idx_1].first*min_x_1, lines_ab[line_idx_1].second,
             min_x_1, min_y_1);
      printf("[svm_struct_custom] quad idx %d: %g*x^2 + %g*x + %g -> (%g, %g + %g + %g) = (%g, %g)\n",
             line_idx_2, quadratic_coeff, lines_ab[line_idx_2].first, lines_ab[line_idx_2].second,
             min_x_2, quadratic_coeff*min_x_2*min_x_2, lines_ab[line_idx_2].first*min_x_2, lines_ab[line_idx_2].second,
             min_x_2, min_y_2);
      printf("[svm_struct_custom] Picked (%g, %g)\n", step_size, min_y);
      // -------------------------
      */

      //printf("q_int_x: %e, min_x_1: %e, min_x_2: %e\n", q_int_x, min_x_1, min_x_2); //debug
      //printf("min_y: %e, min_y_1: %e, min_y_2: %e\n", min_y, min_y_1, min_y_2); //debug

      // pick x with minimum y value
      //step_size = (min_y_1<min_y_2)?min_x_1:min_x_2;

      //printf("[svm_struct_custom] autostep quadratic min (%g,%d)\n", step_size, min_y);

#if CUSTOM_VERBOSITY > 1
      write_scalar("autostep_obj_qc.txt", quadratic_coeff);
      write_scalar("autostep_obj_qt.txt", quadratic_coeff*step_size*step_size);
      write_scalar("autostep_obj_0.txt", min_y);
      write_scalar("autostep_obj_int.txt", min_int_y);
#endif

      if(min_y < -1e-10) { //sanity check
        printf("[svm_struct_custom] Error: min_y is negative!!\n");

        printf("intersections:\n");
        for(int i=0; i<(int)rel_intersections.size(); i++) {
          printf("(%e, %e) ", rel_intersections[i].first, rel_intersections[i].second);
        }
        printf("\n");

        printf("y_values:\n");
        for(int i=0; i<(int)y_values.size(); i++) {
          printf("(%e, %d) ", y_values[i].first, y_values[i].second);
        }
        printf("\n");

        printf("all lines_ab\n");
        for(int i=0; i<(int)line_indices.size(); i++) {
          printf("[%e, %e] ", lines_ab[line_indices[i]].first, lines_ab[line_indices[i]].second);
        }
        printf("\n");

        printf("Exit\n");
        exit(-1);
      }

    }
    break;
  default:
    break;
  }

  step_size *= gparm->shrinkage_coefficient;
  return step_size;
}

double compute_max_slack(STRUCT_LEARN_PARM *sparm, double* w, EXAMPLE *ex, int il, SWORD* fy_to)
{
  ConstraintSet* cs = ConstraintSet::Instance();

  double score_gt = 0;
  SWORD* wy_gt = fy_to;
  while (wy_gt->wnum) {
    score_gt += w[wy_gt->wnum]*wy_gt->weight;
    ++wy_gt;
  }

  double avg_score = 0;
  double avg_loss = 0;
  double avg_slack = 0;
  double max_slack = 0;

  const vector< constraint >* _cs  = cs->getConstraints(ex[il].x.id);
  if(_cs) {
    for(vector<constraint>::const_iterator it = _cs->begin();
        it != _cs->end(); ++it) {
      double score = cs->computeScore(it->first->w, w);
      double loss = it->first->loss;
      double slack = score - score_gt + loss;
      //printf("[ConstraintSet] Score = %g, Score-Score_GT = %g, Loss = %g, Slack = %g\n",
      //       score, score - score_gt, loss, slack);

      avg_score += score;
      avg_loss += loss;
      avg_slack += slack;

      if(slack > max_slack) { max_slack = slack; }
    }
    uint n_cs = _cs->size();
    avg_score /= n_cs;
    avg_loss /= n_cs;
    avg_slack /= n_cs;
  }

  //printf("[svm_struct_custom] Avg_Score = %g, Avg_Score-Score_GT = %g, Avg_Loss = %g, Avg_Slack = %g, Max_Slack = %g\n",
  //       avg_score, avg_score - score_gt, avg_loss, avg_slack, max_slack);
  return max_slack;
}

double compute_max_slack(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                         GRADIENT_PARM* gparm, EXAMPLE *examples, long nExamples, double* w)
{
  double max_slack = 0;
  int _sizePsi = sm->sizePsi + 1;
  SWORD* fy_to = new SWORD[_sizePsi];

  for(int i = 0; i < nExamples; i++) {

    compute_psi_to(sparm, sm, &examples[i], gparm, fy_to);

    double max_slack_i = compute_max_slack(sparm, w, examples, i, fy_to);
    max_slack += max_slack_i;
  }
  max_slack /= nExamples;

  delete[] fy_to;
  return max_slack;
}

double compute_autostep_objective(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm,
                                  GRADIENT_PARM* gparm, EXAMPLE *examples, long nExamples,
                                  double* w, double* dfy, double lambda)
{
  double max_slack = compute_max_slack(sparm, sm, gparm, examples, nExamples, w);
  double sq_norm_dfy = get_sq_norm(dfy, sm->sizePsi+1);
  double quadratic_coeff = nExamples*gparm->autostep_regularization*sq_norm_dfy;
  if(gparm->autostep_regularization_type == REGULARIZATION_PA_TIME_ADAPTIVE) {
    quadratic_coeff *= pow(sparm->iterationId+1, gparm->autostep_regularization_exp);
  }

  quadratic_coeff *= lambda*lambda;

  double obj = quadratic_coeff + max_slack;

  return obj;
}

void update_w(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm, GRADIENT_PARM* gparm,
              double* momentum, double *dfy, double* smw)
{
  int _sizePsi = sm->sizePsi + 1;

  // do not add +1 here as dfy also has an additional dummy entry at index 0.
  // double* smw = sm->w;
  if(momentum) {
    // update momentum
    for(int i = 1; i < _sizePsi; ++i) {
      momentum[i] = (gparm->learning_rate*(dfy[i] + (gparm->regularization_weight*smw[i])) + gparm->momentum_weight*momentum[i]);
    }
    for(int i = 1; i < _sizePsi; ++i) {
      smw[i] -= momentum[i];
    }
  } else {
    for(int i = 1; i < _sizePsi; ++i) {
      smw[i] -= gparm->learning_rate*(dfy[i]+(gparm->regularization_weight*smw[i]));
    }
  }
}

double update_w_with_optimal_step_size(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm, GRADIENT_PARM* gparm,
                                     double* momentum, double *dfy, SWORD* fy_to, EXAMPLE* ex, int il, constraint* c)
{
  stringstream sout;

  /*
  // export one text file per constraint to see if there learning rate oscilates
  sout << "autostep_learning_rate_all_";
  sout << c->first->id;
  sout << ".txt";
  ofstream ofs_cs_autostep_learning_rate(sout.str().c_str(), ios::app);
  */

  ofstream ofs_cs_autostep_learning_rate("autostep_learning_rate_all.txt", ios::app);

  double step_size_bak = gparm->learning_rate;
  double step_size = compute_optimal_step_size(sparm, sm, gparm, ex, il, fy_to, dfy);

  printf("[svm_struct_custom] step_size_bak = %g, step_size = %g\n", step_size_bak, step_size);

  if(fabs(step_size) < gparm->autostep_min_learning_rate) {
    ofstream ofs("unused_step_sizes.txt", ios::app);
    ofs << sparm->iterationId << " " << step_size << endl;
    ofs.close();
    int sign_step_size = (step_size<0)?-1:1;
    step_size = sign_step_size*gparm->autostep_min_learning_rate;
    SSVM_PRINT("[svm_struct_custom] Replacing step size -> %g %d %g\n", step_size, sign_step_size, gparm->autostep_min_learning_rate);
  }

  if(step_size != 0) {
    gparm->learning_rate = step_size;
  } else {
    if(c && (c->first->step_size != 0)) {
      gparm->learning_rate = c->first->step_size;
      SSVM_PRINT("[svm_struct_custom] Selected step size is 0. Move anyway with step size = %g\n", gparm->learning_rate);
    }
  }

  if(gparm->check_slack_after_update) {

    double max_slack_before = compute_max_slack(sparm, sm->w, ex, il, fy_to);

    CUSTOM_VERBOSITY_F(2, ofs_cs_autostep_learning_rate << " " << gparm->learning_rate;)
      
    update_w(sparm, sm, gparm, momentum, dfy, sm->w);

    double max_slack_after = compute_max_slack(sparm, sm->w, ex, il, fy_to);

#if CUSTOM_VERBOSITY > 1
    write_scalar("d_slack.txt", max_slack_after - max_slack_before);
#endif

    // todo: try to multiply by slack??
    if(fabs(max_slack_after - max_slack_before) < 1e-5) {
      if(c && (c->first->step_size != 0)) {
        gparm->learning_rate = c->first->step_size;
      } else {
        gparm->learning_rate = step_size_bak;
        //printf("[svm_struct_custom] gparm->learning_rate = step_size_bak = %g\n", gparm->learning_rate);
      }
      // Slack did not change. Go back to previous learning rate and update.
      SSVM_PRINT("[svm_struct_custom] Slack did not change. Go back to previous learning rate and update. Constraint %d learning_rate %g\n",
                 c->first->id, gparm->learning_rate);

      CUSTOM_VERBOSITY_F(2, ofs_cs_autostep_learning_rate << " " << gparm->learning_rate << endl;)

      update_w(sparm, sm, gparm, momentum, dfy, sm->w);

      // hack to avoid oscillations
      // don't think this happens but just want to be cautious
      ++gparm->n_unchanged_slack;
      if(gparm->n_unchanged_slack > 5) {
        printf("[svm_struct_custom] Warning: slack did not change after 5 iterations.\n");
        c->first->step_size = 1e-9;
        gparm->learning_rate = 1e-9;
        gparm->n_unchanged_slack = 0;

        update_w(sparm, sm, gparm, momentum, dfy, sm->w);
      }

    } else {
      if(c) {
        c->first->step_size = gparm->learning_rate;
      }
      CUSTOM_VERBOSITY_F(2, ofs_cs_autostep_learning_rate << " " << gparm->learning_rate << endl;)
    }
  } else {

    CUSTOM_VERBOSITY_F(2, ofs_cs_autostep_learning_rate << " " << gparm->learning_rate << endl;)

    update_w(sparm, sm, gparm, momentum, dfy, sm->w);
  }

  ofs_cs_autostep_learning_rate.close();

  return step_size;
}

double do_gradient_step(STRUCT_LEARN_PARM *sparm,
                        STRUCTMODEL *sm, EXAMPLE *ex, long nExamples,
                        GRADIENT_PARM* gparm,
                        double* momentum,
                        SWORD* fy_to, SWORD* fy_away, double *dfy,
                        double& dscore,
                        LABEL* y_bar)
{
  int _sizePsi = sm->sizePsi + 1;
  LABEL* y_direct = 0;

  double* _lossPerLabel = sparm->lossPerLabel;
  if(gparm->ignore_loss) {
    sparm->lossPerLabel = 0;
  }

  // setting this to 1 will make the example loop below single thread so that
  // several threads can be run for different temperature while running the
  // samplign code.
#define USE_SAMPLING 0

#if USE_SAMPLING
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
#endif

  /*** precomputation step ***/
  for(int i = 0; i < nExamples; i++) {

#if USE_SAMPLING
#ifdef USE_OPENMP
    int threadId = omp_get_thread_num();
    printf("[svm_struct_custom] Thread %d/%d\n", threadId,omp_get_num_threads());
#endif
#endif

    if(sparm->loss_type == SLACK_RESCALING) {
      y_bar[i] = find_most_violated_constraint_slackrescaling(ex[i].x, ex[i].y,
                                                             sm, sparm);
    } else {
      y_bar[i] = find_most_violated_constraint_marginrescaling(ex[i].x, ex[i].y,
                                                              sm, sparm);
    }
  }

  if(gparm->ignore_loss) {
    sparm->lossPerLabel = _lossPerLabel;
  }

  if(gparm->gradient_type == GRADIENT_DIRECT_ADD ||
     gparm->gradient_type == GRADIENT_DIRECT_SUBTRACT) {

    // allocate memory
    y_direct = new LABEL[nExamples];

    // temporarily remove loss
    double* _lossPerLabel = sparm->lossPerLabel;
    sparm->lossPerLabel = 0;

    for(int il = 0; il < nExamples; il++) {

#ifdef USE_OPENMP
      int threadId = omp_get_thread_num();
      printf("[svm_struct_custom] Thread %d/%d\n", threadId,omp_get_num_threads());
#else
      int threadId = 0;
#endif

      // check if labels are stored in the cache
      int cacheId = nExamples + ex[il].x.id;
      bool labelFound = LabelCache::Instance()->getLabel(cacheId, *y_direct);
      if(!labelFound) {
        // allocate memory
        y_direct->nNodes = ex[il].y.nNodes;
        y_direct->nodeLabels = new labelType[y_direct->nNodes];
        for(int n = 0; n < ex[il].y.nNodes; n++) {
          y_direct->nodeLabels[n] = ex[il].y.nodeLabels[n];
        }
        y_direct->cachedNodeLabels = false;
        labelFound = true;
      }

      runInference(ex[il].x, ex[il].y, sm, sparm, y_direct[il], threadId, labelFound, cacheId);
      //exportLabels(sparm, &ex[il], y_bar, "direct/");

    }
    sparm->lossPerLabel = _lossPerLabel;
  }

#if CUSTOM_VERBOSITY > 2
  ofstream ofs_cs_dscore("dscore.txt", ios::app);
  ofstream ofs_cs_norm_dfy("norm_dfy.txt", ios::app);
  ofstream ofs_cs_norm_w("norm_w.txt", ios::app);
  ofstream ofs_cs_learning_rate("learning_rate.txt", ios::app);

  ofs_cs_norm_w << std::setprecision(12);
#endif

  int n_satisfied = 0;
  int n_not_satisfied = 0;

  const double dfy_weight = 1.0;
  double total_dscore = 0;
  double total_dloss = 0;

  if(gparm->constraint_set_type == CS_USE_MVC) {

    ConstraintSet* cs = ConstraintSet::Instance();

    for(int il = 0; il < nExamples; il++) { /*** example loop ***/

      double _loss = 0;
      compute_gradient(sparm, sm, &ex[il], &y_bar[il], &y_direct[il], gparm,
                       fy_to, fy_away, dfy, &_loss, dfy_weight);

      // add the current constraint first
      if(gparm->constraint_set_type == CS_MARGIN || gparm->constraint_set_type == CS_MARGIN_DISTANCE) {
        double margin = total_dscore + total_dloss;
        double sorting_value = (fabs(margin) < 1e-38)?0 : 1.0/margin;
        cs->add(ex[il].x.id, fy_away, _loss, _sizePsi, sorting_value);
      } else {
        cs->add(ex[il].x.id, fy_away, _loss, _sizePsi);
      }

      const constraint* c = cs->getMostViolatedConstraint(ex[il].x.id, sm->w);
      double dscore_cs = compute_gradient(sm, gparm, fy_to,
                                          c->first->w, dfy, c->first->loss, dfy_weight);
      bool positive_margin = (dscore_cs + c->first->loss) > 0;

      if( (gparm->loss_type != HINGE_LOSS && gparm->loss_type != SQUARE_HINGE_LOSS) || positive_margin) {

        switch(gparm->update_type) {
        case UPDATE_AUTO_STEP:
          {
            update_w_with_optimal_step_size(sparm, sm, gparm, momentum, dfy, fy_to, ex, il, 0);
            break;
          }
        case UPDATE_RESCALE_GRADIENT:
          {
            gparm->learning_rate = gparm->learning_rate_0/get_norm(dfy, _sizePsi);
            update_w(sparm, sm, gparm, momentum, dfy, sm->w);
            break;
          }
        default:
          break;
        }
      }

#if CUSTOM_VERBOSITY > 2
      CUSTOM_VERBOSITY_F(2, ofs_cs_dscore << dscore_cs << " " << c->first->loss << endl;)
      CUSTOM_VERBOSITY_F(2, ofs_cs_norm_dfy << get_norm(dfy, _sizePsi) << endl;)
      CUSTOM_VERBOSITY_F(2, ofs_cs_norm_w << get_norm(sm->w, _sizePsi) << endl;)
      CUSTOM_VERBOSITY_F(2, ofs_cs_learning_rate << gparm->learning_rate << endl;)
#endif

    }

  } else {
    // use all the constraints in the working set instead of the most violated one

    for(int il = 0; il < nExamples; il++) { /*** example loop ***/

      // compute gradient for last generated constraint
      double _loss;
      double _dscore = compute_gradient(sparm, sm, &ex[il], &y_bar[il], &y_direct[il], gparm,
                                        fy_to, fy_away, dfy, &_loss, dfy_weight);

      if(gparm->use_history) {

        // add last generated constraint to the working set
        ConstraintSet* cs = ConstraintSet::Instance();
        if(gparm->constraint_set_type == CS_MARGIN || gparm->constraint_set_type == CS_MARGIN_DISTANCE) {
          double margin = _dscore + _loss;
          double sorting_value = (fabs(margin) < 1e-38)?0 : 1.0/margin;
          cs->add(ex[il].x.id, fy_away, _loss, _sizePsi, sorting_value);
        } else {
          cs->add(ex[il].x.id, fy_away, _loss, _sizePsi);
        }

        if(gparm->use_atomic_updates) {
          vector< constraint >* constraints  = cs->getMutableConstraints(ex[il].x.id);

          if(constraints) {
            for(vector<constraint>::iterator it = constraints->begin();
                it != constraints->end(); ++it) {
              double dscore_cs = compute_gradient(sm, gparm, fy_to,
                                                  it->first->w, dfy, it->first->loss, dfy_weight);
              total_dloss += it->first->loss;
              bool positive_margin = (dscore_cs + it->first->loss) > 0;

              if( (gparm->loss_type != HINGE_LOSS && gparm->loss_type != SQUARE_HINGE_LOSS) || positive_margin) {
                if(gparm->update_type == UPDATE_AUTO_STEP) {
                  update_w_with_optimal_step_size(sparm, sm, gparm, momentum, dfy, fy_to, ex, il, &(*it));
                  printf("Learning rate after update = %g\n", it->first->step_size);
                } else {
                  if(gparm->update_type == UPDATE_RESCALE_GRADIENT) {
                    gparm->learning_rate = gparm->learning_rate_0/get_norm(dfy, _sizePsi);
                  } else {
                    if(gparm->update_type == UPDATE_PASSIVE_AGGRESSIVE) {
                      gparm->learning_rate = (_loss + dscore_cs)/get_sq_norm(dfy, _sizePsi);
                      if(gparm->regularization_weight != 0) {
                        gparm->learning_rate = std::min(gparm->learning_rate, 1.0/((double)gparm->regularization_weight*gparm->n_total_examples));
                      }
                    }
                  }        
                  update_w(sparm, sm, gparm, momentum, dfy, sm->w);
                }
                total_dscore += dscore_cs;
              } else {
                CUSTOM_VERBOSITY_F(2, SSVM_PRINT("[svm_struct_custom] Margin is not positive, delta_score=%g, loss=%g, margin=%g\n",
                                                 dscore_cs, it->first->loss, dscore_cs + it->first->loss);)
              }

              if(positive_margin) {
                ++n_not_satisfied;
              } else {
                ++n_satisfied;
              }

#if CUSTOM_VERBOSITY > 2
              CUSTOM_VERBOSITY_F(2, ofs_cs_dscore << dscore_cs << " " << it->first->loss << endl;)
              CUSTOM_VERBOSITY_F(2, ofs_cs_norm_dfy << get_norm(dfy, _sizePsi) << endl;)
              CUSTOM_VERBOSITY_F(2, ofs_cs_norm_w << get_norm(sm->w, _sizePsi) << endl;)
              CUSTOM_VERBOSITY_F(2, ofs_cs_learning_rate << gparm->learning_rate << endl;)
#endif
            }   
          }
        } else {
          double loss_cs = 0;
          double dscore_cs = compute_gradient_with_history(sparm, sm, ex+il, gparm,
                                                           fy_to, dfy, &loss_cs);
          bool positive_margin = (dscore_cs + loss_cs) > 0;

          if( (gparm->loss_type != HINGE_LOSS && gparm->loss_type != SQUARE_HINGE_LOSS) || positive_margin) {

            switch(gparm->update_type) {
            case UPDATE_AUTO_STEP:
              update_w_with_optimal_step_size(sparm, sm, gparm, momentum, dfy, fy_to, ex, il, 0);
              break;
            case UPDATE_RESCALE_GRADIENT:
              gparm->learning_rate = gparm->learning_rate_0/get_norm(dfy, _sizePsi);
              update_w(sparm, sm, gparm, momentum, dfy, sm->w);
              break;
            default:
              break;
            }

            total_dscore += dscore_cs;
          } else {
            CUSTOM_VERBOSITY_F(2, SSVM_PRINT("[svm_struct_custom] Margin is not positive, score=%g, loss=%g, margin=%g\n",
                                             dscore_cs, loss_cs, dscore_cs+loss_cs);)
            gparm->use_random_weights = true;
          }

          if(positive_margin) {
            ++n_not_satisfied;
          } else {
            ++n_satisfied;
          }

#if CUSTOM_VERBOSITY > 2
          CUSTOM_VERBOSITY_F(2, ofs_cs_dscore << dscore_cs << " " << loss_cs << endl;)
          CUSTOM_VERBOSITY_F(2, ofs_cs_norm_dfy << get_norm(dfy, _sizePsi) << endl;)
          CUSTOM_VERBOSITY_F(2, ofs_cs_norm_w << get_norm(sm->w, _sizePsi) << endl;)
          CUSTOM_VERBOSITY_F(2, ofs_cs_learning_rate << gparm->learning_rate << endl;)
#endif

        }

      } else {

        total_dloss += _loss;

        // update w if constraint is violated
        if( (gparm->loss_type != HINGE_LOSS && gparm->loss_type != SQUARE_HINGE_LOSS) || ((_dscore + _loss) > 0)) {
          total_dscore += _dscore;

          switch(gparm->update_type) {
          case UPDATE_AUTO_STEP:
            update_w_with_optimal_step_size(sparm, sm, gparm, momentum, dfy, fy_to, ex, il, 0);
            break;
          case UPDATE_RESCALE_GRADIENT:
            gparm->learning_rate = gparm->learning_rate_0/get_norm(dfy, _sizePsi);
            update_w(sparm, sm, gparm, momentum, dfy, sm->w);
            break;
          case UPDATE_PASSIVE_AGGRESSIVE:
            printf("[svm_struct_custom] PA\n");
            gparm->learning_rate = (_loss + _dscore)/get_sq_norm(dfy, _sizePsi);
            if(gparm->regularization_weight != 0) {
              gparm->learning_rate = std::min(gparm->learning_rate, 1.0/((double)gparm->regularization_weight*gparm->n_total_examples));
            }
            update_w(sparm, sm, gparm, momentum, dfy, sm->w);
            break;
          case UPDATE_FRANK_WOLFE:
            {
              // compute ws and ls
              double* ws = new double[_sizePsi];
              for(int i = 0; i < _sizePsi; ++i) {
                ws[i] = -dfy[i]/gparm->regularization_weight;
              }
              
              double ls = 0;
              if(!gparm->ignore_loss) {
                ls = _loss;
              }
              
              // compute optimal step size
              double* dw = new double[_sizePsi]; // w_t - ws
              for(int i = 0; i < _sizePsi; ++i) {
                dw[i] = sm->w[i] - ws[i];
              }
              double dp = 0; // dw*w_t
              for(int i = 0; i < _sizePsi; ++i) {
                dp += dw[i]*sm->w[i];
              }
              printf("[svm_struct_custom] FW regularization_weight = %g, dp = %g, ls = %g, cumulated_loss = %g\n",
                     gparm->regularization_weight, dp, ls, gparm->cumulated_loss);
              
              double pd_gap = gparm->regularization_weight*dp - gparm->cumulated_loss + ls;
              gparm->learning_rate = pd_gap / (gparm->regularization_weight*get_sq_norm(dw, _sizePsi));
              printf("[svm_struct_custom] FW Before clipping %g\n", gparm->learning_rate);
              
              write_scalar("fw_gap.txt", pd_gap);
              write_scalar("fw_learning_rate.txt", gparm->learning_rate);
              write_scalar("fw_norm_ws.txt", get_norm(ws, _sizePsi));
              write_scalar("fw_dp.txt", dp);
              write_scalar("fw_loss.txt", _loss);
              write_scalar("fw_cumulated_loss.txt", gparm->cumulated_loss);
              
              // clip learning rate.
              // do not prevent negative learning rate
              if(gparm->learning_rate > 1.0) {
                gparm->learning_rate = 1.0;
              }
              
              printf("[svm_struct_custom] FW After clipping %g\n", gparm->learning_rate);
              
              // update w and cumulated loss
              gparm->cumulated_loss = ((1.0-gparm->learning_rate)*gparm->cumulated_loss) + (gparm->learning_rate*ls);
              
              for(int i = 0; i < _sizePsi; ++i) {
                sm->w[i] = ((1.0-gparm->learning_rate)*sm->w[i]) + (gparm->learning_rate*ws[i]);
              }
              
              delete[] ws;
              delete[] dw;
              
              break;
            }
          default:
            break;
          }

#if CUSTOM_VERBOSITY > 2
         CUSTOM_VERBOSITY_F(2, ofs_cs_dscore << total_dscore << endl;)
         CUSTOM_VERBOSITY_F(2, ofs_cs_norm_dfy << get_norm(dfy, _sizePsi) << endl;)
         CUSTOM_VERBOSITY_F(2, ofs_cs_norm_w << get_norm(sm->w, _sizePsi) << endl;)
         CUSTOM_VERBOSITY_F(2, ofs_cs_learning_rate << gparm->learning_rate << endl;)
#endif // CUSTOM_VERBOSITY

          }
      }     
    }
  }

#if CUSTOM_VERBOSITY > 2
  ofs_cs_dscore.close();
  ofs_cs_norm_dfy.close();
  ofs_cs_norm_w.close();
  ofs_cs_learning_rate.close();
#endif

#if CUSTOM_VERBOSITY > 1

  ofstream ofs_cs_card("constraint_set_card.txt", ios::app);
  ofs_cs_card << n_satisfied << " " << n_not_satisfied << " " << n_satisfied+n_not_satisfied << endl;
  ofs_cs_card.close();

  if(sparm->giType == T_GI_SAMPLING) {
    ofstream ofs_temp("temperature.txt", ios::app);
    ofs_temp << sparm->sampling_temperature_0 << endl;
    ofs_temp.close();
  }

#endif // CUSTOM_VERBOSITY

  dscore = total_dscore;

  double m = compute_m(sparm, sm, ex, nExamples, gparm, y_bar, y_direct, fy_to, fy_away, dfy);

  if(y_direct) {
    delete[] y_direct;
  }

  return m;
}

double compute_m(STRUCT_LEARN_PARM *sparm,
                 STRUCTMODEL *sm, EXAMPLE *ex, long nExamples,
                 GRADIENT_PARM* gparm, LABEL* y_bar, LABEL* y_direct,
                 SWORD* fy_to, SWORD* fy_away, double *dfy)
{
  const double dfy_weight = 1.0;
  double total_loss = 0; // cumulative loss for all examples
  double total_dscore = 0;

  if(gparm->use_history) {

    // use history of constraints
    ConstraintSet* cs = ConstraintSet::Instance();

    for(int il = 0; il < nExamples; il++) { /*** example loop ***/
      const vector< constraint >* constraints = cs->getConstraints(ex[il].x.id);
      if(constraints) {
        for(vector<constraint>::const_iterator it = constraints->begin();
            it != constraints->end(); ++it) {
          double dscore_cs = compute_gradient(sm, gparm, fy_to,
                                              it->first->w, dfy, it->first->loss, dfy_weight);

          // do not add if negative to avoid adding and subtracting values.
          // this score is just logged, not used in any computation.
          if(dscore_cs > 0) {
            total_dscore += dscore_cs;
            total_loss += it->first->loss;
          }
        }
      }
    }
  } else {
    for(int il = 0; il < nExamples; il++) { /*** example loop ***/
      double _loss  = 0;
      total_dscore += compute_gradient(sparm, sm, &ex[il], &y_bar[il],
                                       &y_direct[il], gparm, fy_to, fy_away,
                                       dfy, &_loss, dfy_weight);
      total_loss += _loss;
    }
  }

#if CUSTOM_VERBOSITY > 1
  // loss computed after updating weight vector
  ofstream ofs("loss.txt", ios::app);
  ofs << total_loss << endl;
  ofs.close();

  // dscore computed after updating weight vector
  ofstream ofs_dscore("a_dscore.txt", ios::app);
  ofs_dscore << total_dscore << endl;
  ofs_dscore.close();
#endif

  double m = total_dscore + total_loss;
  return m;
}

/**
 * compute an estimate of the score
 * return max among a subset of sampled superpixels and only compute score for class 0
 */
double compute_score_estimate(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm, GRADIENT_PARM& gparm, EXAMPLE* example)
{
  const int c = 0; // class 0
  labelType* groundTruthLabels = example->y.nodeLabels;
  EnergyParam param;
  sparmToEnergyParam(*sparm, &param);
  Slice_P* slice = example->x.slice;
  GraphInference gi(slice, (const EnergyParam*)&param, sm->w+1, example->x.feature, 0, 0);

  double max_potential = 0;

  string config_tmp;
  double sampling_rate = 0.3;
  if(Config::Instance()->getParameter("sampling_rate", config_tmp)) {
    sampling_rate = atof(config_tmp.c_str()); 
    printf("[svm_struct] sampling_rate = %g\n", sampling_rate);
  }
  bool draw_samples = sampling_rate != 1.0;
  ulong nSupernodes = slice->getNbSupernodes();
  int sid = 0;
  for(int i = 0; i < nSupernodes*sampling_rate; ++i) {

    if(draw_samples) {
      // Select a pixel at random
      sid = rand() * ((double)nSupernodes/(double)RAND_MAX);
    } else {
      sid = i;
    }

    supernode* s = slice->getSupernode(sid);
    vector < supernode* >* lNeighbors = &(s->neighbors);

    double potential = gi.computeUnaryPotential(slice, sid, c);
    for(vector < supernode* >::iterator itN = lNeighbors->begin();
        itN != lNeighbors->end(); itN++) {
      const int c2 = rand()*sparm->nClasses / (double)RAND_MAX;
      double pairwisePotential = gi.computePairwisePotential(slice, s, (*itN),
                                                             c, c2);
      potential += pairwisePotential;
    }

    if(!gparm.ignore_loss) {
      if(c != groundTruthLabels[sid]) {
        // add loss of the ground truth label
        potential += sparm->lossPerLabel[groundTruthLabels[sid]];
      }
    }

    if(potential > max_potential) {
      max_potential = potential;
    }
  }

  return max_potential;
}

void set_sampling_temperature(STRUCT_LEARN_PARM *sparm,
                              STRUCTMODEL *sm,
                              GRADIENT_PARM& gparm,
                              EXAMPLE* ex,
                              long nExamples,
                              double* dscores,
                              double* ddfy)
{

  int schedulingType = 2;
  string config_tmp;
  if(Config::Instance()->getParameter("schedulingType", config_tmp)) {
    schedulingType = atoi(config_tmp.c_str()); 
  }
  printf("[svm_struct_custom] schedulingType=%d\n", schedulingType);

  int nItems = sparm->stepForOutputFiles;

  switch(schedulingType) {
  case 0:
    {
    // try to lower the temperature as much as we can

    double score = compute_score_estimate(sparm, sm, gparm, &ex[0]);

    // compute average dscore
    double avg_dscore = 0;
    for(int i = 0; i < nItems; ++i) {
      avg_dscore += dscores[i];
    }

#if CUSTOM_VERBOSITY > 2
    {
      ofstream ofs_temp("avg_dscore.txt", ios::app);
      ofs_temp << sparm->iterationId << " " << avg_dscore << endl;
      ofs_temp.close();
    }
#endif

    bool exp_is_inf = false;
    while(!exp_is_inf) {
      exp_is_inf = isinf(exp(score/(sparm->sampling_temperature_0/SAMPLING_MUL_COEFF)));
      if(!exp_is_inf) {
        // decrease randomness
        sparm->sampling_temperature_0 /= SAMPLING_MUL_COEFF;
      }
    }
    }
    break;
  case 1:
    {
    // decrease temperature if dscore < 0
    // increase temperature if not enough changes

    double score = compute_score_estimate(sparm, sm, gparm, &ex[0]);
        
    // compute average dscore
    double avg_dscore = 0;
    for(int i = 0; i < nItems; ++i) {
      avg_dscore += dscores[i];
    }

#if CUSTOM_VERBOSITY > 2
    {
      ofstream ofs_temp("avg_dscore.txt", ios::app);
      ofs_temp << sparm->iterationId << " " << avg_dscore << endl;
      ofs_temp.close();
    }
#endif

    if(avg_dscore < 0) {
      bool exp_is_inf = isinf(exp(score/(sparm->sampling_temperature_0/SAMPLING_MUL_COEFF)));
      if(!exp_is_inf && sparm->sampling_temperature_0 > score*1e-10) {
        // decrease randomness
        sparm->sampling_temperature_0 /= SAMPLING_MUL_COEFF;

        // reset labels in the cache to ground-truth
        for(int i = 0; i < nExamples; i++) {
          int cacheId = ex[i].x.id;
          LabelCache::Instance()->setLabel(cacheId, ex[i].y);
        }

#if CUSTOM_VERBOSITY > 2
        ofstream ofs_temp("temperature_change.txt", ios::app);
        ofs_temp << sparm->iterationId << " " << score << " ";
        ofs_temp << score/sparm->sampling_temperature_0 << " " << exp(score/(sparm->sampling_temperature_0));
        ofs_temp << " " << sparm->sampling_temperature_0 << endl;
        ofs_temp.close();
#endif
      } else {

        // decrease sampling rate
        sparm->sampling_rate = max(0.1, sparm->sampling_rate-0.1);

#if CUSTOM_VERBOSITY > 2
        ofstream ofs_temp("sampling_rate.txt", ios::app);
        ofs_temp << sparm->sampling_rate << endl;
        ofs_temp.close();
#endif

      }
    }

    }
    break;
  case 2:
    {
      // decrease temperature
      if(sparm->iterationId % sparm->stepForOutputFiles) {
        double score = compute_score_estimate(sparm, sm, gparm, &ex[0]);
        bool exp_is_inf = isinf(exp(score/(sparm->sampling_temperature_0/SAMPLING_MUL_COEFF)));
        if(!exp_is_inf) {
          // decrease randomness
          sparm->sampling_temperature_0 /= SAMPLING_MUL_COEFF;
        }
      }
    }
    break;
  default:
    // do not change the temperature
    break;
  }
}

// use -w 9 to call this function
void svm_learn_struct_joint_custom(SAMPLE sample, STRUCT_LEARN_PARM *sparm,
				   LEARN_PARM *lparm, KERNEL_PARM *kparm, 
				   STRUCTMODEL *sm)
     /* Input: sample (training examples)
	       sparm (structural learning parameters)
               lparm (svm learning parameters)
               kparm (kernel parameters)
	       Output: sm (learned model) */
{
  long        nTotalExamples = sample.n;
  EXAMPLE     *examples = sample.examples;

  CONSTSET    cset;
  int         cached_constraint = 0;
  int         numIt = 0;
  double      *alpha = NULL;
  double ceps;
  bool finalized = false;
  string config_tmp;

  init_struct_model(sample, sm, sparm, lparm, kparm); 

  Config* config = Config::Instance();
  GRADIENT_PARM gparm;
  init_gradient_param(gparm, config, ConstraintSet::Instance());
  gparm.examples_all = examples;
  gparm.n_total_examples = nTotalExamples;

  // 0 = initialize parameter vector to 0
  // 1 = initialize parameter vector using random values
  int init_type = INIT_WEIGHT_0;
  if(gparm.ignore_loss == 1) {
    init_type = INIT_WEIGHT_RANDOM;
  }
  if(config->getParameter("sgd_init_type", config_tmp)) {
    init_type = atoi(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] init_type = %d\n", init_type);

  sm->svm_model = 0;
  //sm->w = new double[sm->sizePsi+1];
  // use C style to be compatible with svm-light
  // and to make sure there is no error in free_struct_model
  sm->w = (double *)my_malloc(sizeof(double)*(sm->sizePsi+1));
  init_w(sparm, sm, &gparm, examples, init_type);

  if(sparm->giType == T_GI_SAMPLING) {
    init_sampling(sparm, sm, examples, nTotalExamples);
  }

  int n_iterations_update_learning_rate = 1000;
  if(config->getParameter("n_iterations_update_learning_rate", config_tmp)) {
    n_iterations_update_learning_rate = atoi(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] n_iterations_update_learning_rate = %d\n", n_iterations_update_learning_rate);

  if(gparm.ignore_loss) {
    sparm->lossPerLabel = 0;
  }

  if(gparm.update_type == UPDATE_AUTO_STEP) {

    gparm.autostep_regularization_type = REGULARIZATION_PA;
    if(config->getParameter("sgd_autostep_regularization_type", config_tmp)) {
      gparm.autostep_regularization_type = (eAutoStepRegularizationType)atoi(config_tmp.c_str());
    }
    printf("[SVM_struct_custom] autostep_regularization_type = %d\n", (int)gparm.autostep_regularization_type);

    gparm.autostep_obj_type = OBJECTIVE_ONE_EXAMPLE;
    if(config->getParameter("sgd_autostep_objective_type", config_tmp)) {
      gparm.autostep_obj_type = (eAutoStepObjectiveType)atoi(config_tmp.c_str());
    }
    printf("[SVM_struct_custom] autostep_objective_type = %d\n", (int)gparm.autostep_obj_type);

    double autostep_min_learning_rate = 0;
    if(config->getParameter("autostep_min_learning_rate", config_tmp)) {
      autostep_min_learning_rate = atof(config_tmp.c_str());
    }
    printf("[SVM_struct_custom] autostep_min_learning_rate = %g\n", autostep_min_learning_rate);
    gparm.autostep_min_learning_rate = autostep_min_learning_rate;

    double autostep_reg_exp = 0.5;
    if(config->getParameter("sgd_autostep_reg_exp", config_tmp)) {
      autostep_reg_exp = atof(config_tmp.c_str());
    }
    printf("[SVM_struct_custom] sgd_autostep_reg_exp = %g\n", autostep_reg_exp);
    gparm.autostep_regularization_exp = autostep_reg_exp;

    double autostep_regularization_weight = 0;
    if(config->getParameter("sgd_autostep_regularization_weight", config_tmp)) {
      autostep_regularization_weight = atof(config_tmp.c_str());
    }
    printf("[SVM_struct_custom] autostep_regularization_weight = %g\n", autostep_regularization_weight);
    gparm.autostep_regularization = autostep_regularization_weight;

    if(gparm.autostep_regularization_type == REGULARIZATION_NONE) {
      gparm.autostep_regularization = 0;
    }

    if(gparm.autostep_regularization_type != REGULARIZATION_NONE && autostep_regularization_weight == 0) {
      printf("autostep_regularization_weight has to be different from 0 with regularization mode %d\n", (int)gparm.autostep_regularization_type);
      exit(-1);
    }

    double norm_psi_gt = get_norm_psi_gt(sparm, sm, examples, nTotalExamples);
    printf("[SVM_struct_custom] norm_psi_gt = %g\n", norm_psi_gt);
    if(norm_psi_gt != 0) {
      gparm.autostep_regularization /= norm_psi_gt;
    }
    printf("[SVM_struct_custom] gparm.autostep_regularization = %g\n", gparm.autostep_regularization);
  }

  double* momentum = 0;
  if( (gparm.update_type == UPDATE_MOMENTUM) || (gparm.update_type == UPDATE_MOMENTUM_DECREASING)) {
    int _sizePsi = sm->sizePsi + 1;
    momentum = new double[_sizePsi];
    for(int i = 0; i < _sizePsi; ++i) {
      momentum[i] = 0;
    }
  }

  double last_obj = 0;
  int _sizePsi = sm->sizePsi + 1;
  SWORD* fy_to = new SWORD[_sizePsi];
  SWORD* fy_away = new SWORD[_sizePsi];
  double* dfy = new double[_sizePsi];
  memset((void*)dfy, 0, sizeof(double)*(_sizePsi));
  LABEL* y_bar = new LABEL[nTotalExamples];

  int nMaxIterations = (sparm->nMaxIterations!=-1)?sparm->nMaxIterations:1e7;
  int nItems = sparm->stepForOutputFiles;
  double* ms = new double[nItems];
  double* objs = new double[nItems];
  double* dscores = new double[nItems];
  int idx = 0;
  int example_id = 0;

  // sampling: keep previous dfy and check how much change
  // occurs to decide if we should raise the temperature
  double* dfy_p = 0;
  double* ddfy = 0;
  if(sparm->giType == T_GI_SAMPLING) {
    dfy_p = new double[_sizePsi];
    memset((void*)dfy_p, 0, sizeof(double)*(_sizePsi));
    ddfy = new double[nItems];
  }

  // backup
  double* w_t = (double *)my_malloc(sizeof(double)*(_sizePsi));
  double* w_tp = (double *)my_malloc(sizeof(double)*(_sizePsi)); // perturbed version

  do {

    EXAMPLE* _ex = examples;

    int _nBatchExamples = nTotalExamples;

    if(gparm.n_batch_examples != -1) {
      _ex += example_id;

      // make sure next batch doesn't go over size of training set
      if(example_id + gparm.n_batch_examples > nTotalExamples) {
        _nBatchExamples = nTotalExamples - example_id;
        printf("[SVM_struct_custom] Batch size for iteration %d is too large. Reducing to %d\n", sparm->iterationId, _nBatchExamples);
      } else {
        _nBatchExamples = gparm.n_batch_examples;
      }
    }

    double m = do_gradient_step(sparm, sm, _ex, _nBatchExamples,
                                &gparm, momentum, fy_to, fy_away, dfy,
                                dscores[idx], y_bar);

    // projection
    if(gparm.enforce_submodularity) {
      double* smw = sm->w + 1;
      bool enforced_submodularity = enforce_submodularity(sparm, smw);

#if CUSTOM_VERBOSITY > 1
      if(enforced_submodularity) {
        write_scalar("enforced_submodularity.txt", sparm->iterationId);
      }
#endif

    }
    if(gparm.max_norm_w > 0) {
      project_w(sm, &gparm);
    }
    if(gparm.weight_constraint_type  == POSITIVE_WEIGHTS) {
      truncate_w(sparm, sm, &gparm);
    }

#if CUSTOM_VERBOSITY > 1
    // check slack for all the examples
    double max_slack = compute_max_slack(sparm, sm, &gparm, examples, gparm.n_total_examples, sm->w);
    write_scalar("max_slack.txt", max_slack);

    // compute autostep objective function
    if(gparm.update_type == UPDATE_AUTO_STEP) {
      double obj = compute_autostep_objective(sparm, sm, &gparm, examples, gparm.n_total_examples, sm->w, dfy, gparm.learning_rate);
      write_scalar("autostep_obj.txt", obj);

      //----------------------------

      double learning_rate_bak = gparm.learning_rate;

      // copy new w to w_t
      for(int i = 0; i < _sizePsi; ++i) {
        w_tp[i] = w_t[i];
      }
      gparm.learning_rate = learning_rate_bak + 1e-2*learning_rate_bak;
      update_w(sparm, sm, &gparm, momentum, dfy, w_tp);
      obj = compute_autostep_objective(sparm, sm, &gparm, examples, gparm.n_total_examples, w_tp, dfy, gparm.learning_rate);
      write_scalar("autostep_obj_plus.txt", obj);

      for(int i = 0; i < _sizePsi; ++i) {
        w_tp[i] = w_t[i];
      }
      gparm.learning_rate = learning_rate_bak - 1e-2*learning_rate_bak;
      update_w(sparm, sm, &gparm, momentum, dfy, w_tp);
      obj = compute_autostep_objective(sparm, sm, &gparm, examples, gparm.n_total_examples, w_tp, dfy, gparm.learning_rate);
      write_scalar("autostep_obj_minus.txt", obj);

      gparm.learning_rate = learning_rate_bak;

      // copy new w to w_t
      for(int i = 0; i < _sizePsi; ++i) {
        w_t[i] = sm->w[i];
      }

      //----------------------------

    }
#endif

    // compute objective function
    double obj = 0;
    switch(gparm.loss_type) {
    case LINEAR_LOSS:
      obj = m;
      break;
    case LOG_LOSS:
      if(m > 100) {
        obj = log(1 + std::exp(100.0));
      } else {
        obj = log(1 + std::exp(m));
      }
      break;
    case HINGE_LOSS:
      obj = max(0.0, m);
      break;
    case SQUARE_HINGE_LOSS:
      obj = max(0.0, m*m);
      break;
    default:
      printf("[svm_struct_custom] Unknown loss type %d\n", gparm.loss_type);
      exit(-1);
      break;
    }

    if(numIt == 0) {
      ceps = fabs(obj);
    } else {
      ceps = fabs(last_obj - obj);
    }
    last_obj = obj;

    finalized = finalize_iteration(ceps,cached_constraint,sample,sm,cset,alpha,sparm);

    switch(gparm.update_type)
      {
      case UPDATE_DECREASING:
      case UPDATE_MOMENTUM_DECREASING:
        {
          if(numIt > n_iterations_update_learning_rate) {
            gparm.learning_rate = gparm.learning_rate_0/(double)pow(numIt-n_iterations_update_learning_rate,
                                                                    gparm.learning_rate_exponent);
          } else {
            if(numIt > n_iterations_update_learning_rate) {
              gparm.learning_rate = gparm.learning_rate_0/(double)pow(numIt-n_iterations_update_learning_rate,
                                                                      gparm.learning_rate_exponent);
            }
          }
        }
        break;
      default:
        break;
      }

    ms[idx] = m;
    objs[idx] = obj;

    if(ddfy) {
      // compute difference between successive dfy vectors
      ddfy[idx] = 0;
      for(int i = 0; i < _sizePsi; ++i) {
        ddfy[idx] += fabs(dfy_p[i] - dfy[i]);
        dfy_p[i] = dfy[i];
      }
    }

    // dump stats
    if(idx == nItems-1) {
      idx = 0;
#if CUSTOM_VERBOSITY > 1
      write_scalars("m.txt", ms, nItems);
      write_scalars("obj.txt", objs, nItems);
#endif

      if(momentum) {
        stringstream sout;
        sout << "step_" << nItems << ".txt";
        string s_it = sout.str();
        string momentum_filename = "momentum_" + s_it;
        write_vector(momentum_filename.c_str(), momentum, _sizePsi);
      }

      // dynamically change temperature for sampling
      if(sparm->giType == T_GI_SAMPLING) {
        set_sampling_temperature(sparm, sm, gparm, examples, gparm.n_total_examples, dscores, ddfy);
      }

    } else {
      ++idx;
    }

    fflush(stdout);

    example_id += gparm.n_batch_examples;
    if(example_id >= gparm.n_total_examples) {
      example_id = 0;
    }

    ++numIt;

  } while( (numIt < nMaxIterations) &&
           (!finalized)
	 );

  ConstraintSet::Instance()->save("constraint_set.txt");

  if(numIt >= nMaxIterations) {
    printf("[svm_struct_custom] Reached max number of iterations %d\n",
           nMaxIterations);
  }
  if (ceps <= sparm->epsilon) {
    printf("[svm_struct_custom] Reached epsilon value %g/%g\n",
           ceps, sparm->epsilon);
  }

  if(momentum) {
    delete[] momentum;
    momentum = 0;
  }

  delete[] fy_to;
  delete[] fy_away;
  delete[] dfy;
  delete[] objs;
  delete[] ms;
  delete[] dscores;

  delete[] w_t;
  delete[] w_tp;

  for(int s = 0; s < nTotalExamples; ++s) {
    free_label(y_bar[s]);
  }
  delete[] y_bar;

  if(dfy_p) {
    delete[] dfy_p;
  }
  if(ddfy) {
    delete[] ddfy;
  }
}

void init_sampling(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm, EXAMPLE* examples, long nExamples)
{
  string config_tmp;
  // set temperature for sampling
  double norm_wy_to = get_norm_psi_gt(sparm, sm, examples, nExamples);

  sparm->sampling_temperature_0 = norm_wy_to*1e4;
  printf("[SVM_struct_custom] sampling_temperature_0 = %g\n", sparm->sampling_temperature_0);

  sparm->sampling_rate = 0.3;
  if(Config::Instance()->getParameter("sampling_rate", config_tmp)) {
    sparm->sampling_rate = atof(config_tmp.c_str()); 
  }
  printf("[SVM_struct_custom] sampling_rate = %g\n", sparm->sampling_rate);

  //sparm->lossScale = norm_wy_to;
  sparm->lossScale = 1;
  if(Config::Instance()->getParameter("sgd_loss_scale", config_tmp)) {
    double lossScale = atof(config_tmp.c_str());
    if(lossScale > 0) {
      sparm->lossScale = lossScale;
    }
  }
  printf("[SVM_struct_custom] sparm->lossScale = %g\n", sparm->lossScale);

  for(int c = 0; c < sparm->nClasses; ++c) {
    sparm->lossPerLabel[c] *= sparm->lossScale;
  }
}

void init_w(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm, GRADIENT_PARM* gparm, EXAMPLE* examples, int init_type)
{
  vector<string> paramFiles;
  getFilesInDir("parameter_vector0/", paramFiles, "txt", true);

  if(paramFiles.size() > 0) {
    string initial_parameter_vector = paramFiles[paramFiles.size()-1].c_str();
    printf("[SVM_struct_custom] Loading parameter file %s\n",
           initial_parameter_vector.c_str());
    sm->w[0] = 0;
    EnergyParam _param(initial_parameter_vector.c_str());
    for(int i = 1; i < sm->sizePsi+1; ++i) {
      sm->w[i] = _param.weights[i-1];
    }
    rename(initial_parameter_vector.c_str(), "initial_parameter_vector.txt");
  } else {

    switch(init_type) {
    case INIT_WEIGHT_0:
      {
        // easier to debug cause we know exactly what to expect for the first
        // most violated constraint.
        printf("[SVM_struct_custom] Initializing weight vector to 0\n");
        for(int i = 0; i < sm->sizePsi+1; ++i) {
          sm->w[i] = 0;
        }
        break;
      }
    case INIT_WEIGHT_1:
      {
        printf("[SVM_struct_custom] Initializing unary weights to 1\n");
        int  featIndex0 = SVM_FEAT_INDEX0(sparm);
        sm->w[0] = 0;
        sm->w[1] = 0; // offset
        // pairwise term
        for(int i = 2; i < featIndex0 + 1; ++i) {
          sm->w[i] = 0;
        }
        // set unary weights to -1
        for(int i = featIndex0 + 1; i < sm->sizePsi+1; ++i) {
          sm->w[i] = -1;
        }

        // Computing maximum possible loss for each image
        double avg_loss = 0;
        long nExamples = gparm->n_total_examples;
        for(long idx = 0; idx < nExamples; idx++) {
          for(int n = 0; n < examples[idx].y.nNodes; n++) {    
            avg_loss += sparm->lossPerLabel[examples[idx].y.nodeLabels[n]];
          }
        }
        avg_loss /= nExamples;
        SSVM_PRINT("[SVM_struct] Rescaling initial weight vector with average loss for %ld examples = %g\n", nExamples, avg_loss);

        double norm_w = sqrt((double)(sm->sizePsi - featIndex0));
        double scale = avg_loss/norm_w;
        for(int i = featIndex0 + 1; i < sm->sizePsi+1; ++i) {
          sm->w[i] *= scale;
        }

        break;
      }
    case INIT_WEIGHT_RANDOM:
    default:
      {
        printf("[SVM_struct_custom] Initializing weight vector randomly\n");
        srand(time(NULL));
        double norm_w = 0;
        sm->w[0] = 0;
        for(int i = 1; i < sm->sizePsi+1; ++i) {
          sm->w[i] = rand() / (double)RAND_MAX;
          norm_w += sm->w[i];
        }

        double initial_norm_w = -1;
        string config_tmp;
        if(Config::Instance()->getParameter("initial_norm_w", config_tmp)) {
          initial_norm_w = atof(config_tmp.c_str());
        }
        printf("[SVM_struct_custom] initial_norm_w = %g\n", initial_norm_w);

        if(initial_norm_w > 0 && norm_w > 0) {
          double scale = initial_norm_w/norm_w;
          printf("[SVM_struct_custom] Rescaling w by %g\n", scale);
          for(int i = 1; i < sm->sizePsi+1; ++i) {
            sm->w[i] *= scale;
          }
        }
      }
      break;
    }

    // projection
    if(gparm->enforce_submodularity) {
      double* smw = sm->w + 1;
      enforce_submodularity(sparm, smw);
    }

    stringstream sout;
    sout << "parameter_vector0";
    sout << "/iteration_0.txt";
    save_parameters(sout.str().c_str(), sparm, sm);

  }
}

void truncate_w(STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm, GRADIENT_PARM* gparm)
{
  for(int i = 1; i <= sparm->nUnaryWeights; ++i) {
    if(sm->w[i] > 0) {
      sm->w[i] = 0;
    }
  }

  /*
  double min_w = sm->w[1];
  for(int i = 2; i < sm->sizePsi+1; ++i) {
    if(sm->w[i] < min_w) {
      min_w = sm->w[i];
    }
  }

  if(min_w < 0) {
    for(int i = 1; i < sm->sizePsi+1; ++i) {
      sm->w[i] -= min_w;
    }
  }
  */
}

void project_w(STRUCTMODEL *sm, GRADIENT_PARM* gparm)
{
  double norm_w = 0;
  for(int i = 1; i < sm->sizePsi+1; ++i) {
    norm_w += sm->w[i];
  }

  if(gparm->max_norm_w > 0 && norm_w > gparm->max_norm_w) {
    double scale = gparm->max_norm_w/norm_w;
    printf("[SVM_struct_custom] Rescaling w by %g/%g = %g\n",
           gparm->max_norm_w, norm_w, scale);
    for(int i = 1; i < sm->sizePsi+1; ++i) {
      sm->w[i] *= scale;
    }
  }
}

bool enforce_submodularity(STRUCT_LEARN_PARM *sparm, double* smw)
{
  bool enforced_submodularity = false;
  double* pw = smw + sparm->nUnaryWeights;

  int nloops = 1;
  int offset = 0;
  if(sparm->nClasses > 2) {
    nloops = 2;
    offset = 1;
  }

  // For 2 classes, use graph-cuts if pairwise potential is attractive/submodular.
  // Let E be the energy and S the score (E=-S)
  // Recall that potential is submodular if :
  // E(0,0) + E(1,1) =< E(0,1) + E(1,0)
  // S(0,0) + S(1,1) >= S(0,1) + S(1,0) or d >= a

#if USE_LONG_RANGE_EDGES
  bool submodularEnergy = isSubmodular(sparm,smw);
#endif

  for(int l = 0 ; l < nloops; ++l) {

    double d = 0; // diagonal
    double a = 0; //anti-diagonal
    double d2 = 0; // diagonal
    double a2 = 0; //anti-diagonal

#if USE_LONG_RANGE_EDGES
    for(int p = 0; submodularEnergy && (p < sparm->nDistances); ++p)
      {
        a = 0;
        d = 0;
#else
        {
#endif

          for(int g = 0; g < sparm->nGradientLevels; ++g) {
            d += pw[0] + pw[3+offset];
            a += pw[1] + pw[2+offset];
            if(a > d) {
              // compute accumulated terms up to gradient - 1
              a2 = 0;
              d2 = 0;
              double* pw2 = smw + sparm->nUnaryWeights;
              if(l > 0) {
                pw2 += 4;
              }
              for(int g2 = 0; g2 < g; ++g2) {
                d2 += pw2[0] + pw2[3+offset];
                a2 += pw2[1] + pw2[2+offset];
                pw2 += sparm->nClasses*sparm->nClasses;
              }

              // make sure that a < d (remember that submodular means that the sum of the terms on the diagonal is higher)
              pw[1] = (d-a2)/2.0 - 1e-2;
              pw[2+offset] = pw[1];

              // add last term
              d2 += pw2[0] + pw2[3+offset];
              a2 += pw2[1] + pw2[2+offset];
          
              a = a2;
              d = d2;
              enforced_submodularity = true;
            } else {
              if(sparm->useSymmetricWeights) {
                // a <= d
                // make sure matrix is symmetric
                pw[1] = (pw[1] + pw[2+offset])/2.0;
                pw[2+offset] = pw[1];
              }
            }

            pw += sparm->nClasses*sparm->nClasses;
          }
        }

        pw = smw + sparm->nUnaryWeights;
        pw += 4;
      }

    return enforced_submodularity;
}

void init_gradient_param(GRADIENT_PARM& gparm, Config* config,
                         ConstraintSet* constraint_set)
{
  string config_tmp;

  double learning_rate = 1e-9;
  if(config->getParameter("learning_rate", config_tmp)) {
    learning_rate = atof(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] learning_rate = %g\n", learning_rate);

  eUpdateType sgd_update_type = UPDATE_CONSTANT;
  if(config->getParameter("sgd_update_type", config_tmp)) {
    sgd_update_type = (eUpdateType)atoi(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] sgd_update_type = %d\n", (int)sgd_update_type);

  float momentum_weight = 0;
  if(config->getParameter("sgd_momentum_weight", config_tmp)) {
    momentum_weight = atof(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] momentum_weight = %g\n", momentum_weight);
  assert(momentum_weight <= 1.0);

  float regularization_weight = 0;
  if(config->getParameter("sgd_regularization_weight", config_tmp)) {
    regularization_weight = atof(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] regularization_weight = %g\n", regularization_weight);

  if(sgd_update_type == UPDATE_FRANK_WOLFE && regularization_weight == 0) {
    printf("[SVM_struct_custom] Regularization weight should be different from 0 for Frank-Wolfe algorithm.");
    exit(-1);
  }

  double learning_rate_exponent = 0.50;
  if(config->getParameter("learning_rate_exponent", config_tmp)) {
    learning_rate_exponent = atof(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] learning_rate_exponent = %g\n", learning_rate_exponent);

  bool update_loss_function = false;
  if(config->getParameter("update_loss_function", config_tmp)) {
    update_loss_function = config_tmp.c_str()[0] == '1';
  }
  printf("[SVM_struct_custom] update_loss_function=%d\n", (int)update_loss_function);

  eLossType sgd_loss_type = HINGE_LOSS;
  if(config->getParameter("sgd_loss_type", config_tmp)) {
    sgd_loss_type = (eLossType)atoi(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] sgd_loss_type = %d\n", (int)sgd_loss_type);
  assert(sgd_loss_type != LINEAR_LOSS);

  eGradientType sgd_gradient_type = GRADIENT_GT;
  if(config->getParameter("sgd_gradient_type", config_tmp)) {
    sgd_gradient_type = (eGradientType)atoi(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] sgd_gradient_type = %d\n", (int)sgd_gradient_type);

  int sgd_n_batch_examples = -1; // use all examples at once
  if(config->getParameter("sgd_n_batch_examples", config_tmp)) {
    sgd_n_batch_examples = atoi(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] sgd_n_batch_examples = %d\n", sgd_n_batch_examples);

  int max_number_constraints = 10000;
  bool sgd_use_history = true;
  if(config->getParameter("cs_max_number_constraints", config_tmp)) {
    max_number_constraints = atoi(config_tmp.c_str());
    sgd_use_history = max_number_constraints > 0;
  }
  printf("[SVM_struct_custom] sgd_use_history = %d\n", (int)sgd_use_history);

  eSortingType constraint_set_type = CS_DISTANCE;
  if(config->getParameter("constraint_set_type", config_tmp)) {
    constraint_set_type = (eSortingType)atoi(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] constraint_set_type = %d\n", (int)constraint_set_type);
  constraint_set->setSortingAlgorithm(constraint_set_type);

  bool sgd_ignore_loss = false;
  if(config->getParameter("sgd_ignore_loss", config_tmp)) {
    sgd_ignore_loss = config_tmp.c_str()[0] == '1';
  }
  printf("[SVM_struct_custom] sgd_ignore_loss = %d\n", (int)sgd_ignore_loss);

  double max_norm_w = -1;
  if(Config::Instance()->getParameter("max_norm_w", config_tmp)) {
    max_norm_w = atof(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] max_norm_w = %g\n", max_norm_w);

  bool enforce_submodularity = true;
  if(Config::Instance()->getParameter("project_w", config_tmp)) {
    enforce_submodularity = config_tmp.c_str()[0] == '1';
  }
  if(Config::Instance()->getParameter("enforce_submodularity", config_tmp)) {
    enforce_submodularity = config_tmp.c_str()[0] == '1';
  }
  printf("[SVM_struct_custom] enforce_submodularity = %d\n", (int)enforce_submodularity);

  bool use_atomic_updates = true;
  if(Config::Instance()->getParameter("use_atomic_updates", config_tmp)) {
    use_atomic_updates = config_tmp.c_str()[0] == '1';
  }
  printf("[SVM_struct_custom] use_atomic_updates = %d\n", (int)use_atomic_updates);

  bool use_random_weights = false;
  if(Config::Instance()->getParameter("use_random_weights", config_tmp)) {
    use_random_weights = config_tmp.c_str()[0] == '1';
  }
  printf("[SVM_struct_custom] use_random_weights = %d\n", (int)use_random_weights);
  if(use_random_weights) {
    printf("[SVM_struct_custom] srand\n");
    //srand(time(NULL));
    struct timeval _t;
    gettimeofday(&_t, NULL);
    srand(_t.tv_usec);
  }

  double shrinkage_coefficient = 1.0;
  if(Config::Instance()->getParameter("shrinkage_coefficient", config_tmp)) {
    shrinkage_coefficient = atof(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] shrinkage_coefficient = %g\n", shrinkage_coefficient);

  bool check_slack_after_update = true;
  if(Config::Instance()->getParameter("check_slack_after_update", config_tmp)) {
    check_slack_after_update = config_tmp.c_str()[0] == '1';
  }
  printf("[SVM_struct_custom] check_slack_after_update = %d\n", (int)check_slack_after_update);

  eWeightConstraintType weight_constraint_type = NO_WEIGHT_CONSTRAINTS;
  if(config->getParameter("weight_constraint_type", config_tmp)) {
    weight_constraint_type = (eWeightConstraintType)atoi(config_tmp.c_str());
  }
  printf("[SVM_struct_custom] weight_constraint_type = %d\n", (int)weight_constraint_type);

  gparm.learning_rate = learning_rate;
  gparm.learning_rate_0 = learning_rate; // initial learning rate
  gparm.learning_rate_exponent = learning_rate_exponent;
  gparm.max_norm_w = max_norm_w;
  gparm.enforce_submodularity = enforce_submodularity;
  gparm.update_type = sgd_update_type;
  gparm.loss_type = sgd_loss_type;
  gparm.momentum_weight = momentum_weight;
  gparm.regularization_weight = regularization_weight;
  gparm.gradient_type = sgd_gradient_type;
  gparm.use_history = sgd_use_history;
  gparm.constraint_set_type = constraint_set_type;
  gparm.ignore_loss = sgd_ignore_loss;
  gparm.n_batch_examples = sgd_n_batch_examples;
  gparm.use_atomic_updates = use_atomic_updates;
  gparm.use_random_weights = use_random_weights;
  gparm.shrinkage_coefficient = shrinkage_coefficient;
  gparm.check_slack_after_update = check_slack_after_update;
  gparm.weight_constraint_type = weight_constraint_type;
}
