/************************************************************************/
/* (c) 2012-2013 Ecole Polytechnique Federale de Lausanne               */
/* All rights reserved.                                                 */
/*                                                                      */
/* EPFL grants a non-exclusive and non-transferable license for non     */
/* commercial use of the Software for education and research purposes   */
/* only. Any other use of the Software is expressly excluded.           */
/*                                                                      */
/* Redistribution of the Software in source and binary forms, with or   */
/* without modification, is not permitted.                              */
/*                                                                      */
/* Written by Carlos Becker                                             */
/*                                                                      */
/* http://cvlab.epfl.ch/software/synapse                                */
/* Contact <carlos.becker@epfl.ch> for comments & bug reports.          */
/************************************************************************/

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <itkImage.h>
#include <itkVectorImage.h>
#include <itkIndex.h>
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkNthElementImageAdaptor.h"

#include "SymmetricEigenAnalysisTest.h"

#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSymmetricEigenAnalysisImageFilter2.h"

#include "itkComposeImageFilter.h"
#include "itkMatrixIndexSelectionImageFilter.h"

#include "SingleEigenVectorOfHessian2.h"
#include "itkImageDuplicator.h"


// Macro to avoid re-typing with itk
// --> For instance: 
//			makeNew( random, itk::RandomImageSource<FloatImage2DType> );
// is equivalent to:
//			itk::RandomImageSource<FloatImage2DType>::Pointer	random;
//			random = itk::RandomImageSource<FloatImage2DType>::New();
// The __VA_ARGS__ is there so that it can handle commas within the argument list 
//   in a natural way
#define makeNew(instanceName, ...)	\
    __VA_ARGS__::Pointer instanceName = __VA_ARGS__::New()


//////////////////////////////////////////////////////////////////////
// I love typedefing my data

//Image
const int Dimension = 3;
typedef unsigned char                                   PixelType;
typedef itk::Image<PixelType, Dimension>          ImageType;
typedef ImageType::IndexType                      IndexType;
typedef itk::ImageFileReader< ImageType >         ReaderType;

typedef itk::FixedArray<float, Dimension>         OrientationPixelType;
typedef itk::Image<OrientationPixelType, Dimension>
OrientationImageType;
typedef itk::ImageFileReader< OrientationImageType >     OrientationImageReaderType;
typedef itk::ImageFileWriter< ImageType >     FloatImageWriterType;


typedef itk::ImageRegionConstIterator< OrientationImageType > ConstOrientationIteratorType;
typedef itk::ImageRegionIterator< OrientationImageType>       OrientationIteratorType;

typedef itk::ImageRegionConstIterator< ImageType > ConstFloatIteratorType;
typedef itk::ImageRegionIterator< ImageType>       FloatIteratorType;


/** Hessian & utils **/
typedef itk::SymmetricSecondRankTensor<float,Dimension>                       HessianPixelType;
typedef itk::Image< HessianPixelType, Dimension >                             HessianImageType;
typedef itk::HessianRecursiveGaussianImageFilter<ImageType, HessianImageType> HessianFilterType;

typedef itk::Vector<float, Dimension>            VectorPixelType;

typedef itk::Vector<float, Dimension>          EigenValuePixelType;
typedef itk::Matrix<float, Dimension, Dimension>	EigenVectorPixelType;

typedef itk::VectorImage<float, Dimension> VectorImageType;

typedef itk::Image<EigenValuePixelType, Dimension> EigenValueImageType;
typedef itk::Image<EigenVectorPixelType, Dimension> EigenVectorImageType;
typedef EigenValueImageType 		FirstEigenVectorOrientImageType;


typedef itk::ComposeImageFilter< ImageType, VectorImageType >	ComposeFilterType;
typedef itk::MatrixIndexSelectionImageFilter< EigenVectorImageType, ImageType >	MatrixIndexSelectionFilterType;


typedef itk::SymmetricEigenAnalysisImageFilter2< HessianImageType, EigenValueImageType, EigenVectorImageType, FirstEigenVectorOrientImageType >        HessianToEigenFilter;

using namespace std;

int SingleEigenVectorOfHessian2Execute(float sigma, const itk::Image<unsigned char, 3>::Pointer image,
                                       std::string outputFile, WhichEigVec whichEig, float zAnisotropyFactor)
{
	//  typename RotationalFeatureType::Pointer featureRotational
	//    = RotationalFeatureType::New();

	//////////////////////////////////////////////////////////////////////
	// Reads the image

    std::cout << "Reading the image" << std::endl;

    typedef itk::ImageDuplicator< ImageType > DuplicatorType;
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(image);
    duplicator->Update();
    ImageType::Pointer inpImg = duplicator->GetOutput();

    
    ImageType::SpacingType spacing = inpImg->GetSpacing();
    spacing[2] = zAnisotropyFactor;
    
    std::cout << "Using spacing: " << spacing << ", anisotr factor = " << zAnisotropyFactor << std::endl;
    inpImg->SetSpacing(spacing);
    
    std::cout << "Hessian filtering" << std::endl;
    makeNew( hessianFilt, HessianFilterType );
    hessianFilt->SetSigma( sigma );
    hessianFilt->SetInput( inpImg );

    std::cout << "Computing eigenvalues" << std::endl;
    // now compute eigenvalues/main eigenvector
    makeNew( eigenFilt, HessianToEigenFilter );
    
    eigenFilt->SetGenerateEigenVectorImage(true);
    eigenFilt->SetGenerateFirstEigenVectorOrientImage(false);
    
    // only sort by magnitude if highest magnitude is required
    eigenFilt->SetOrderEigenValues( whichEig != EHighestMagnitude );
    eigenFilt->SetOrderEigenMagnitudes( whichEig == EHighestMagnitude );
    
    eigenFilt->SetInput( hessianFilt->GetOutput() );

#if 0
    typedef itk::VectorResampleImageFilter< VectorImageType, VectorImageType > VectorResampleFilterType;
    makeNew( vecResampler, VectorResampleFilterType );

    vecResampler->SetInput(eigenFilt->GetOutput());
    vecResampler->SetSize(image->GetLargestPossibleRegion().GetSize());
#endif

	makeNew( matSelFilt1, MatrixIndexSelectionFilterType );
	makeNew( matSelFilt2, MatrixIndexSelectionFilterType );
	makeNew( matSelFilt3, MatrixIndexSelectionFilterType );
	
	if ( whichEig == EHighestMagnitude || whichEig == EHighest )
    {
        matSelFilt1->SetIndices( 2, 0 );
        matSelFilt2->SetIndices( 2, 1 );
        matSelFilt3->SetIndices( 2, 2);
	} 
	else 
	{
        matSelFilt1->SetIndices( 0, 0 );
        matSelFilt2->SetIndices( 0, 1 );
        matSelFilt3->SetIndices( 0, 2 );
	}
	
	matSelFilt1->SetInput( eigenFilt->GetEigenVectorImage() );
	matSelFilt2->SetInput( eigenFilt->GetEigenVectorImage() );
	matSelFilt3->SetInput( eigenFilt->GetEigenVectorImage() );

    makeNew( composeFilt, ComposeFilterType );
    composeFilt->SetInput(0, matSelFilt1->GetOutput() );
    composeFilt->SetInput(1, matSelFilt2->GetOutput() );
    composeFilt->SetInput(2, matSelFilt3->GetOutput() );


   std::cout << "Saving images" << std::endl;
    {
        composeFilt->Update();
        VectorImageType::Pointer outImg = composeFilt->GetOutput();
        composeFilt->Update();
        
        // reset spacing
        spacing[0] = spacing[1] = spacing[2] = 1.0;
        
        outImg->SetSpacing( spacing );
        
        
        makeNew( writer, itk::ImageFileWriter<VectorImageType> );
        writer->SetInput( outImg );
        writer->SetFileName( outputFile );
        writer->Update();
    }

    return EXIT_SUCCESS;
}
