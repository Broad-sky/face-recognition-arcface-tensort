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
include CMakeFiles/myplugins.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/myplugins.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/myplugins.dir/flags.make

CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o: CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o.depend
CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o: CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o.Debug.cmake
CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o: ../prelu.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/sps/face_recognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o"
	cd /home/sps/face_recognition/build/CMakeFiles/myplugins.dir && /usr/bin/cmake -E make_directory /home/sps/face_recognition/build/CMakeFiles/myplugins.dir//.
	cd /home/sps/face_recognition/build/CMakeFiles/myplugins.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/sps/face_recognition/build/CMakeFiles/myplugins.dir//./myplugins_generated_prelu.cu.o -D generated_cubin_file:STRING=/home/sps/face_recognition/build/CMakeFiles/myplugins.dir//./myplugins_generated_prelu.cu.o.cubin.txt -P /home/sps/face_recognition/build/CMakeFiles/myplugins.dir//myplugins_generated_prelu.cu.o.Debug.cmake

# Object files for target myplugins
myplugins_OBJECTS =

# External object files for target myplugins
myplugins_EXTERNAL_OBJECTS = \
"/home/sps/face_recognition/build/CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o"

libmyplugins.so: CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o
libmyplugins.so: CMakeFiles/myplugins.dir/build.make
libmyplugins.so: /usr/local/cuda-10.2/lib64/libcudart.so
libmyplugins.so: CMakeFiles/myplugins.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sps/face_recognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libmyplugins.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/myplugins.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/myplugins.dir/build: libmyplugins.so

.PHONY : CMakeFiles/myplugins.dir/build

CMakeFiles/myplugins.dir/requires:

.PHONY : CMakeFiles/myplugins.dir/requires

CMakeFiles/myplugins.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/myplugins.dir/cmake_clean.cmake
.PHONY : CMakeFiles/myplugins.dir/clean

CMakeFiles/myplugins.dir/depend: CMakeFiles/myplugins.dir/myplugins_generated_prelu.cu.o
	cd /home/sps/face_recognition/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sps/face_recognition /home/sps/face_recognition /home/sps/face_recognition/build /home/sps/face_recognition/build /home/sps/face_recognition/build/CMakeFiles/myplugins.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/myplugins.dir/depend

