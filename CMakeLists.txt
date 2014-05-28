cmake_minimum_required(VERSION 2.4)

PROJECT(sliceMe)

set(SLICEME_DIR ${CMAKE_SOURCE_DIR})

include(CMakeLists_common.txt)

####################################################################### LIBRARY

add_library(superpixel
${SLICEME_FILES}
)

TARGET_LINK_LIBRARIES(superpixel ${OPENCV_LIBRARIES} ${SUPERPIXEL_LIBRARY})
if(UNIX)
	SET_TARGET_PROPERTIES(superpixel PROPERTIES COMPILE_FLAGS -fPIC)
endif(UNIX)
SET_TARGET_PROPERTIES(superpixel PROPERTIES ENABLE_EXPORTS TRUE)


###################################################################### BINARIES
SET_SOURCE_FILES_PROPERTIES( ${SLICEME_DIR}/core/train.c ${INFERENCE_FILES} ${SLICEME_FILES} PROPERTIES LANGUAGE CXX )
SET_SOURCE_FILES_PROPERTIES( ${SLICEME_DIR}/core/getopt_long.c ${SLICEME_DIR}/core/getopt.h ${SLICEME_DIR}/core/getopt.c PROPERTIES LANGUAGE C )

if(WIN32)
set(INFERENCE_FILES_WIN32
${SLICEME_DIR}/core/gettimeofday.c
${SLICEME_DIR}/core/getopt.c
${SLICEME_DIR}/core/getopt_long.c
)
endif()

ADD_EXECUTABLE(train
${SLICEME_DIR}/core/train.c
${INFERENCE_FILES}
${SLICEME_FILES}
${INFERENCE_FILES_WIN32}
)
TARGET_LINK_LIBRARIES(train ${SLICEME_THIRD_PARTY_LIBRARIES})

ADD_EXECUTABLE(predict
${SLICEME_DIR}/core/predict.cpp
${INFERENCE_FILES}
${SLICEME_FILES}
${INFERENCE_FILES_WIN32}
)
TARGET_LINK_LIBRARIES(predict ${SLICEME_THIRD_PARTY_LIBRARIES})

set(BUILD_PREDICT_MBEST ON CACHE BOOL "build also predict mbest (only supported in linux)")
if(BUILD_PREDICT_MBEST)
	ADD_EXECUTABLE(predict_mbest
	${SLICEME_DIR}/core/predict_mbest.cpp
	${INFERENCE_FILES}
	${SLICEME_FILES}
	)
	TARGET_LINK_LIBRARIES(predict_mbest ${SLICEME_THIRD_PARTY_LIBRARIES})
endif(BUILD_PREDICT_MBEST)
