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

// standard libraries
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <fstream>
#include <algorithm>

// SliceMe
#include "Config.h"
#include "oSVM.h"
#include "utils.h"
#include "globalsE.h"

using namespace std;

//------------------------------------------------------------------------------

void oSVM::initSVMNode(osvm_node*& x, int d)
{
  x = new osvm_node[d+1];
  int i = 0;
  for(i = 0; i < d; i++)
    x[i].index = i+1;
  x[i].index = -1;
}

void oSVM::initSVMNodes(osvm_node**& xs, uint nNodes, int d)
{
  xs = new osvm_node*[nNodes];
  for(uint k=0; k < nNodes; k++)
    {
      xs[k] = new osvm_node[d+1];
      int i = 0;
      for(i = 0;i < d; i++)
        xs[k][i].index = i+1;
      xs[k][i].index = -1;
    }
}

void oSVM::print(osvm_node *x, const char* title)
{
  if(title)
    printf("%s: \n", title);
  for(int i = 0;x[i].index != -1; i++) {
    printf("%d:%g ",x[i].index,x[i].value);
  }
  printf("\n");
}

void oSVM::printNonZeros(osvm_node *x, const char* title)
{
  if(title)
    printf("%s: \n", title);
  for(int i = 0;x[i].index != -1; i++) {
    if (x[i].value != 0) {
      printf("%d:%g ",x[i].index,x[i].value);
    }
  }
  printf("\n");
}

double oSVM::norm(osvm_node* x, int norm_type)
{
  double _norm = 0;
  if(norm_type == L1_NORM) {
    for(int i = 0;x[i].index != -1; i++) {
      _norm += x[i].value;
    }
  } else {
    for(int i = 0;x[i].index != -1; i++) {
      _norm += x[i].value*x[i].value;
    }
    _norm = sqrt(_norm);
  }
  return _norm;
}
