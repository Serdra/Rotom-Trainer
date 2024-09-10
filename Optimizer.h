#pragma once
#include "includes.h"

class ADAM{
   public:
    static constexpr float b1 = 0.9;
    static constexpr float b2 = 0.999;
    static constexpr float epsilon = 1e-8;

    NNUEGrad v;
    NNUEGrad s;

    ADAM() {};

    void step(NNUEGrad grad, float lr) {
        // Updates the ADAM values for each parameter and the parameters themselves
        for(int i = 0; i < 18; i++) {
            for(int j = 0; j < 768; j++) {
                for(int k = 0; k < neurons; k++) {
                    v.hiddenWeightGradient[i][j][k] = (v.hiddenWeightGradient[i][j][k] * b1) + ((1 - b1) * grad.hiddenWeightGradient[i][j][k]);
                    s.hiddenWeightGradient[i][j][k] = (s.hiddenWeightGradient[i][j][k] * b2) + ((1 - b2) * grad.hiddenWeightGradient[i][j][k] * grad.hiddenWeightGradient[i][j][k]);
                    hiddenWeights[i][j][k] -= lr * (v.hiddenWeightGradient[i][j][k] / sqrt(s.hiddenWeightGradient[i][j][k] + epsilon));
                }
            }
        }
        for(int i = 0; i < neurons; i++) {
            v.hiddenBiasGradient[i] = (v.hiddenBiasGradient[i] * b1) + ((1 - b1) * grad.hiddenBiasGradient[i]);
            s.hiddenBiasGradient[i] = (s.hiddenBiasGradient[i] * b2) + ((1 - b2) * grad.hiddenBiasGradient[i] * grad.hiddenBiasGradient[i]);
            hiddenBias[i] -= lr * (v.hiddenBiasGradient[i] / sqrt(s.hiddenBiasGradient[i] + epsilon));
        }

        for(int i = 0; i < neurons*2; i++) {
            v.outputWeightGradient[i] = (v.outputWeightGradient[i] * b1) + ((1 - b1) * grad.outputWeightGradient[i]);
            s.outputWeightGradient[i] = (s.outputWeightGradient[i] * b2) + ((1 - b2) * grad.outputWeightGradient[i] * grad.outputWeightGradient[i]);
            outputWeights[i] -= lr * (v.outputWeightGradient[i] / sqrt(s.outputWeightGradient[i] + epsilon));
        }

        v.outputBiasGradient = (v.outputBiasGradient * b1) + ((1 - b1) * grad.outputBiasGradient);
        s.outputBiasGradient = (s.outputBiasGradient * b2) + ((1 - b2) * grad.outputBiasGradient * grad.outputBiasGradient);
        outputBias -= lr * (v.outputBiasGradient / sqrt(s.outputBiasGradient + epsilon));
    }
};