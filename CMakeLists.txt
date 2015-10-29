project(FaceDetection)
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
FIND_PACKAGE(Boost COMPONENTS system filesystem program_options REQUIRED)
INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/include
        include/
        ${PROJECT_SOURCE_DIR}/lib/
        lib/
        ${CMAKE_CURRENT_BINARY_DIR}
)

SET(NEEDED_LIBS
        ${OpenCV_LIBS}
    	${Boost_CHRONO_LIBRARY}
	${Boost_FILESYSTEM_LIBRARY}
	${Boost_DATE_TIME_LIBRARY}
	${Boost_IOSTREAMS_LIBRARY}
	${Boost_SERIALIZATION_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
	${Boost_THREAD_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

SET(LIB_SRC_LIST modelfile.cpp asmmodel.cpp shapevec.cpp similaritytrans.cpp afreader.cpp modelimage.cpp  shapemodel.cpp shapeinfo.cpp)

file(GLOB SOURCE_FILES
	${PROJECT_SOURCE_DIR}/src/*.cpp
)

file(GLOB HEADER_FILES
        ${PROJECT_SOURCE_DIR}/include/FaceDetection.h
)

SET(HEADER_FILES
        include/FaceDetection.h
)

SET(SOURCE_FILES
        src/main.cpp
        src/FaceDetection.cpp
)

ADD_SUBDIRECTORY(lib)
ADD_EXECUTABLE(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} asm ${NEEDED_LIBS})
