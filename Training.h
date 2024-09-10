#pragma once
#include "includes.h"

// Training hyperparameters
const int batchSize = 16384;
const float resultPercent = 1.0;
float lr = 1e-3;
float lr_decay = 1;

float runningAverageDecay = 0.999;

// Grabs the next sample to compute the gradient for and calculates it until the batch is finished
void threadedGradCalculator(TrainingPosition* data, NNUEGrad &grad, float &loss, int &next, std::mutex &mtx);

// Computes the gradient over an entire batch by spawning threads and then summing the results
std::pair<NNUEGrad, float> batchGradient(DataLoader &loader);

// Main training loop
void train(DataLoader &loader, int maxBatches, int printInterval);