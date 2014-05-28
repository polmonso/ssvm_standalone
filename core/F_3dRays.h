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

#ifndef F_3DRAYS_H
#define F_3DRAYS_H

#include <cv.h>
#include "rays3d.h"

// SliceMe
#include "Feature.h"
#include "Slice3d.h"

class F_3dRays : public Feature
{
 public:	

  F_3dRays(uchar* _rawData,
           int _nx, int _ny, int _nz,
           uchar lowerTh = 8,
           uchar upperTh = 16);

  ~F_3dRays();

  int getSizeFeatureVectorForOneSupernode();

  void init();

 protected:
  bool getFeatureVectorForOneSupernode(osvm_node *x, Slice3d* slice3d, int supernodeId);

  bool getFeatureVectorForOneSupernode(osvm_node *n, Slice3d* slice3d,
                                       const int x, const int y, const int z);

 private:
  vector<Rays3d*> rays3d;

  uchar* rawData;
  int nx;
  int ny;
  int nz;

  bool rotationInvariantVector;
  int fvSize;

  eRayCoreType rayCoreType;
  int rayFeaturesType;
  bool use2DConnectedComponents;
  int minComponentSize;
  string paramSurfaceFilename;

  
  uchar lowerTh_start;
  uchar lowerTh_step;
  uchar lowerTh_end;
  uchar upperTh_start;
  uchar upperTh_step;
  uchar upperTh_end;
  
};

#endif // F_3DRAYS_H
