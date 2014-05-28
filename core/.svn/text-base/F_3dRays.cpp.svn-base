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

#include "Config.h"
#include "F_3dRays.h"
#include "utils.h"

F_3dRays::F_3dRays(uchar* _rawData,
                   int _nx, int _ny, int _nz,
                   uchar lowerTh,
                   uchar upperTh)
{
  rawData = _rawData;
  nx = _nx;
  ny = _ny;
  nz = _nz;
  init();
}

void F_3dRays::init()
{
  rayCoreType = PS_ICOSAHEDRON;
  string paramPlatonicSurfaceType;
  if(Config::Instance()->getParameter("rays3d_surfaceType", paramPlatonicSurfaceType)) {
    rayCoreType = (eRayCoreType)atoi(paramPlatonicSurfaceType.c_str());
    PRINT_MESSAGE("[F_3dRays] paramPlatonicSurfaceType=%d\n", (int)rayCoreType);
  }

  minComponentSize = 800;
  string paramMinComponentSize;
  if(Config::Instance()->getParameter("rays3d_minComponentSize", paramMinComponentSize)) {
    minComponentSize = atoi(paramMinComponentSize.c_str());
    PRINT_MESSAGE("[F_3dRays] minComponentSize=%d\n", minComponentSize);
  }

  use2DConnectedComponents = false;
  string paramUse2DConnectedComponents;
  if(Config::Instance()->getParameter("rays3d_use2DConnectedComponents", paramUse2DConnectedComponents)) {
    if(paramUse2DConnectedComponents.c_str()[0] == '1')
      use2DConnectedComponents = true;
    PRINT_MESSAGE("[F_3dRays] use2DConnectedComponents=%d\n", (int)use2DConnectedComponents);
  }

  rayFeaturesType = FT_DISTANCE;
  string paramRayFeaturesType;
  if(Config::Instance()->getParameter("rays3d_rayFeaturesType", paramRayFeaturesType)) {
    rayFeaturesType = atoi(paramRayFeaturesType.c_str());
    PRINT_MESSAGE("[F_3dRays] rayFeaturesType=%d\n", rayFeaturesType);
  }

  if(Config::Instance()->getParameter("rays3d_surfaceFilename", paramSurfaceFilename)) {
    PRINT_MESSAGE("[F_3dRays] paramSurfaceFilename=%s\n", paramSurfaceFilename.c_str());        
  } else {
    printf("[F_3dRays] Error : rays3d_surfaceFilename should be specified in the config file\n");
    exit(-1);
  }

  string paramRotationInvariantVector;
  Config::Instance()->getParameter("rotationInvariantVector", paramRotationInvariantVector);
  if((paramRotationInvariantVector == "1")||(paramRotationInvariantVector == "true")) {
    rotationInvariantVector = true;
    PRINT_MESSAGE("[F_3dRays] rotationInvariantVector=true\n");
  } else {
    rotationInvariantVector = false;
    PRINT_MESSAGE("[F_3dRays] rotationInvariantVector=false\n");
  }

  string paramLowerTh;
  if(Config::Instance()->getParameter("rays3d_lowerTh", paramLowerTh)) {
    lowerTh_start = (uchar)atoi(paramLowerTh.c_str());
    PRINT_MESSAGE("[F_3dRays] lowerTh=%d\n",(int)lowerTh_start);
  } else {
    lowerTh_start = 8; // default
  }
  lowerTh_end = lowerTh_start;
  lowerTh_step = 2;

  string paramUpperTh;
  if(Config::Instance()->getParameter("rays3d_upperTh", paramUpperTh)) {
    upperTh_start = (uchar)atoi(paramUpperTh.c_str());
    PRINT_MESSAGE("[F_3dRays] upperTh=%d\n",(int)upperTh_start);
  }
  upperTh_end = upperTh_start;
  upperTh_step = 2;

  float gaussianVariance = 10.0f;
  string paramGaussianVariance;
  if(Config::Instance()->getParameter("rays3d_gaussianVariance", paramGaussianVariance)) {
    gaussianVariance = atof(paramGaussianVariance.c_str());
    PRINT_MESSAGE("[F_3dRays] gaussianVariance=%f\n", gaussianVariance);
  }
 
  // loop over all parameters and create instances of rays3d
  for(float g = gaussianVariance; g <= gaussianVariance; g += 8) {
  //for(float g = gaussianVariance - 8; g <= gaussianVariance; g += 8) {
    for(uchar l = lowerTh_start; l <= lowerTh_end; l += lowerTh_step) {
      for(uchar u = upperTh_start; u <= upperTh_end; u += upperTh_step) {

        Rays3d* _rays3d = 0;
        if(rayCoreType == PS_LOADFROMFILE) {
          _rays3d = new Rays3d(rawData, nx, ny, nz,
                               rayFeaturesType,
                               paramSurfaceFilename.c_str());
        } else {
          _rays3d = new Rays3d(rawData, nx, ny, nz,
                               rayFeaturesType,
                               rayCoreType);
        }
      
        rays3d.push_back(_rays3d);
        _rays3d->computeCannyCube(l, u, g,
                                  minComponentSize,
                                  use2DConnectedComponents);
      }
    }
  }

  fvSize = 0;
  for(vector<Rays3d*>::iterator it = rays3d.begin(); it != rays3d.end(); ++it) {
    fvSize += (*it)->getSizeFeatureVector();
  }
  PRINT_MESSAGE("[F_3dRays] Feature vector size = %d\n", fvSize);
}

F_3dRays::~F_3dRays()
{
  for(vector<Rays3d*>::iterator it = rays3d.begin(); it != rays3d.end(); ++it) {
    delete *it;
  }
}

bool F_3dRays::getFeatureVectorForOneSupernode(osvm_node *x, Slice3d* slice3d, int supernodeId)
{
  supernode* s = slice3d->getSupernode(supernodeId);

  for(int i = 0; i < fvSize; i++) {
    x[i].value = 0;
  }

  int stepSize = ceil(s->size()/(float)NB_SAMPLED_PIXELS);
  node n;
  nodeIterator ni = s->getIterator();
  ni.goToBegin();
  while(!ni.isAtEnd()) {
    ni.get(n);
    for(int i = 0; i < stepSize; i++) {
      ni.next();
      if(ni.isAtEnd())
        break;
    }
    
    for(vector<Rays3d*>::iterator it = rays3d.begin(); it != rays3d.end(); ++it) {
      (*it)->getFeatureVector(n.x,n.y,n.z,
                              (feature_node*)x,
                              rotationInvariantVector);
    }
  }

  for(int i = 0; i < fvSize; i++) {
    x[i].value = x[i].value/NB_SAMPLED_PIXELS;
  }

  return true;
}

bool F_3dRays::getFeatureVectorForOneSupernode(osvm_node *n, Slice3d* slice3d,
                                const int x, const int y, const int z)
{
  for(int i = 0; i < fvSize; i++) {
    n[i].value = 0;
  }

  for(vector<Rays3d*>::iterator it = rays3d.begin(); it != rays3d.end(); ++it) {
    (*it)->getFeatureVector(x,y,z,
                            (feature_node*)n,
                            rotationInvariantVector);
  }
 
  return true;
}

int F_3dRays::getSizeFeatureVectorForOneSupernode()
{
  return (int)fvSize;
}
