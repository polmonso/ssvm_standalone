Code structure
==============
* core contains the core classes
* lib contains the third party libraries used by the code
* tests contains implementation of algorithms
* tools contains different applications like SVM training, pixel/superpixel/supervoxel-based classification, graphcuts, ssvm...
* roc contains scripts to generate ROCs from prediction files (those files contains probabilities for each node or edge)

Dependencies
============
* opencv-dev
* libsvm (modified to be multi-thread)
* ITK (install with review flag ON)
* freeglut3-dev libglew1.5-dev libxmu-dev
* sudo apt-get install g++ make cmake doxygen graphviz libboost-dev libboost-graph-dev libboost-program-options-dev
* gnuplot

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

2. Compile third-party libraries :
  1. Go to slic and type "cmake .; make"
  2. Go to libDAI024 and type "cmake .; make"
  3. ITK : Download from the web site and use ccmake to set review flag to ON.

3. Main code

```
mkdir build
cd build
ccmake ..
make -j
```

You can edit CMakeLists_common.txt or just change the flags with the ccmake interface.
You might want to turn off some of the dependencies. Look at the USE_??? flags (you might need to toggle advanced mode pressing `t` key).

Deprecated instructions
=======================

4. Tools
41. Go to tools/whatevertoolyouwant/ and type "cmake .; make"

SSVM
For ssvm, edit svm_struct_globals.h and change NTHREADS
Running the code
- Check folders "groundTruth" and "predictionLabels" and make sure the images look OK.
- The code will output results every 10 iterations or every time it updates the C value.

5. Options :
Make sure $LOCALHOME/src/EM/superpixels/config.txt is present. It contains some of the options to change for one database. For VOC, voc=1 should be set and voc=0 for MSRC.

---------------------------------------------------------------------------------


bashrc :
if [ `hostname` == 'cvlabpc44' ]; then
    export PATH=$PATH:~/src/EM/neurons/c++/bin:~/src/EM/Cpp/steerableFilters2D/bin:~/usr/bin/liblinear-1.33:~/bin/EM44:~/bin:~/src/EM/superpixels/lib/libsvm-3.0/
    export LOCALHOME=/localhome/aurelien/
    export OMP_NUM_THREADS=16
elif [ `hostname` == 'cvlabpc45' ]; then
    export PATH=$PATH:~/src/EM/neurons/c++/bin:~/src/EM/Cpp/steerableFilters2D/bin:~/usr/bin/liblinear-1.33:~/bin/EM45:~/bin:~/src/EM/superpixels/lib/libsvm-3.0/
    export LOCALHOME=/localhome/aurelien/
    export OMP_NUM_THREADS=16
else
    export PATH=$PATH:~/src/EM/neurons/c++/bin:~/src/EM/Cpp/steerableFilters2D/bin:~/usr/bin/liblinear-1.33:~/bin/EM:~/bin:~/src/EM/superpixels/lib/libsvm-3.0/
    export OMP_NUM_THREADS=8
fi
