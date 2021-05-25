#!/bin/bash

cython --cplus wikinew/*.pyx
cython --cplus wikinew/utils/*.pyx
cython --cplus wikinew/utils/tokenizer/*.pyx
cython --cplus wikinew/utils/sentence_detector/*.pyx
