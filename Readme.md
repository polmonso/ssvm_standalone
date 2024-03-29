Code structure
==============
* core contains the core classes
* lib contains the third party libraries used by the code
<!-- * tests contains implementation of algorithms -->
<!-- * tools contains different applications like SVM training, pixel/superpixel/supervoxel-based classification, graphcuts, ssvm... -->
<!-- * roc contains scripts to generate ROCs from prediction files (those files contains probabilities for each node or edge) -->

Dependencies
============
* opencv-dev
* libsvm (modified to be multi-thread)
* ITK 3.20.1(install with review flag ON)
* freeglut3-dev libglew1.5-dev libxmu-dev
* sudo apt-get install g++ make cmake doxygen graphviz libboost-dev libboost-graph-dev libboost-program-options-dev
* gnuplot

Windows will have to at least download cmake-gui, opencv and boost.


Provided Dependencies
---------------------
* libDAI
* slic
* auxiliary ITK classes

Sources
=======

clone with

`git clone https://git.epfl.ch/repo/ssvm.git`

Install :

1. Install the code in $LOCALHOME/src/EM/superpixels/ or $HOME/src/EM/superpixels/

2. Compile third-party libraries (linux):
  1. Go to slic/build and type `cmake .; make` (you might have to create the build directory)
  2. Go to libDAI024/build and type `cmake ..; make`
  3. ITK : Download version 3.20.1 from the web sitei and use `ccmake ..` to set review flag to ON. Then build

2. Compile third-party libraries (windows):
  1. Go to slic/build and type `cmake-gui ..` and then `MSBuild.exe supervoxel.sln /p:Configuratio=Release /m`
  2. Go to libDAI024/build and type `cmake-gui ..`
  3. ITK : Download version 3.20.1 from the web site and use `cmake-gui` to set review flag to ON. Then build.

3. Main code

```
mkdir build
cd build
ccmake ..
make -j
```

You can edit `CMakeLists_common.txt` or just change the flags with the ccmake interface.
You might want to turn off some of the dependencies. Look at the `USE_???` flags (you might need to toggle advanced mode pressing `t` key).

Test volumes
============

You can download a test volume and config file from [dataset](https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/testdata.zip) and [config file](./sampledata/sample_config.txt). From the directory where the binaries are located, we would run:

`./train sample_config.txt`
`./predict -c sample_config.txt -w parameter_vector0/iteration_400.txt -v 1 -g 12`
`./predict -c sample_config.txt -w parameter_vector0/iteration_400.txt -v 1 -g 12 > output_predict.txt 2>&1`

change the paths where images are located accordingly on the config to run the algorithm on your dataset.

Troubleshooting
===============

On windows, when building slic/superpixels might give the error "No Target Architecture". Solve by adding the definition `SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_AMD64_")` to the CMakeLists.txt



