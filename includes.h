#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <fstream>
#include "string.h"
#include <sstream>
#include <x86intrin.h>
#include <thread>
#include <omp.h>
#include <mutex>

#include "PokeChess/PokeChess.h"
using namespace pokechess;

#include "NNUE.h"
#include "Optimizer.h"
#include "Dataloader.h"
#include "Training.h"

#include "Dataloader.cpp"
#include "Training.cpp"