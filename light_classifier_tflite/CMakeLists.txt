cmake_minimum_required(VERSION 3.5)
# esp_lcd component triggers GCC ICE on this toolchain version; we don't use it.
set(EXCLUDE_COMPONENTS "esp_lcd")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(light_classifier_tflite)
