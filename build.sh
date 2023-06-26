#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra -ggdb -I./thirdparty/ `pkg-config --cflags raylib`"
LIBS="-lm `pkg-config --libs raylib` -lglfw -ldl -lpthread"

clang $CFLAGS -o img2nn_rgb img2nn_rgb.c $LIBS
