# CMake generated Testfile for 
# Source directory: /home/spoluri/opencv/modules/flann
# Build directory: /home/spoluri/opencv/build_judge/modules/flann
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_flann "/home/spoluri/opencv/build_judge/bin/opencv_test_flann" "--gtest_output=xml:opencv_test_flann.xml")
set_tests_properties(opencv_test_flann PROPERTIES  LABELS "Main;opencv_flann;Accuracy" WORKING_DIRECTORY "/home/spoluri/opencv/build_judge/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/spoluri/opencv/cmake/OpenCVUtils.cmake;1799;add_test;/home/spoluri/opencv/cmake/OpenCVModule.cmake;1365;ocv_add_test_from_target;/home/spoluri/opencv/cmake/OpenCVModule.cmake;1123;ocv_add_accuracy_tests;/home/spoluri/opencv/modules/flann/CMakeLists.txt;2;ocv_define_module;/home/spoluri/opencv/modules/flann/CMakeLists.txt;0;")
