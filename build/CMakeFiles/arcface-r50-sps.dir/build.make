# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sps/face_recognition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sps/face_recognition/build

# Include any dependencies generated for this target.
include CMakeFiles/arcface-r50-sps.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/arcface-r50-sps.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/arcface-r50-sps.dir/flags.make

CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o: CMakeFiles/arcface-r50-sps.dir/flags.make
CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o: ../arcface-r50-sps.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sps/face_recognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o -c /home/sps/face_recognition/arcface-r50-sps.cpp

CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sps/face_recognition/arcface-r50-sps.cpp > CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.i

CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sps/face_recognition/arcface-r50-sps.cpp -o CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.s

CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o.requires:

.PHONY : CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o.requires

CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o.provides: CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o.requires
	$(MAKE) -f CMakeFiles/arcface-r50-sps.dir/build.make CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o.provides.build
.PHONY : CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o.provides

CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o.provides.build: CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o


# Object files for target arcface-r50-sps
arcface__r50__sps_OBJECTS = \
"CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o"

# External object files for target arcface-r50-sps
arcface__r50__sps_EXTERNAL_OBJECTS =

arcface-r50-sps: CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o
arcface-r50-sps: CMakeFiles/arcface-r50-sps.dir/build.make
arcface-r50-sps: libmyplugins.so
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
arcface-r50-sps: /usr/local/cuda-10.2/lib64/libcudart.so
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
arcface-r50-sps: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
arcface-r50-sps: CMakeFiles/arcface-r50-sps.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sps/face_recognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable arcface-r50-sps"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/arcface-r50-sps.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/arcface-r50-sps.dir/build: arcface-r50-sps

.PHONY : CMakeFiles/arcface-r50-sps.dir/build

CMakeFiles/arcface-r50-sps.dir/requires: CMakeFiles/arcface-r50-sps.dir/arcface-r50-sps.cpp.o.requires

.PHONY : CMakeFiles/arcface-r50-sps.dir/requires

CMakeFiles/arcface-r50-sps.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/arcface-r50-sps.dir/cmake_clean.cmake
.PHONY : CMakeFiles/arcface-r50-sps.dir/clean

CMakeFiles/arcface-r50-sps.dir/depend:
	cd /home/sps/face_recognition/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sps/face_recognition /home/sps/face_recognition /home/sps/face_recognition/build /home/sps/face_recognition/build /home/sps/face_recognition/build/CMakeFiles/arcface-r50-sps.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/arcface-r50-sps.dir/depend
