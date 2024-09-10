#include "includes.h"

// This is written in kind of a weird way, the idea is to multithread it later
void threadedGradCalculator(TrainingPosition* data, NNUEGrad &grad, float &loss, int &next, std::mutex &mtx) {
    while(true) {
        // Gets the next example to calculate
        mtx.lock();
        if(next == batchSize) {
            mtx.unlock();
            return;
        }
        int idx = next;
        next++;
        mtx.unlock();

        // 0 is accumulator from white perspective, 1 is from black
        float acc[2][neurons];
        float output = outputBias;

        // Applies the bias
        memcpy(acc[0], hiddenBias, sizeof(float) * neurons);
        memcpy(acc[1], hiddenBias, sizeof(float) * neurons);

        // Goes through each feature and calculates the accumulator
        for(int i = 0; i < 32; i++) {
            // 0 would correspond to a pawn on a1 which is impossible, allowing us to skip once it is encountered
            if(data[idx].features[i] == 0) break;

            for(int j = 0; j < neurons; j++) {
                acc[0][j] += hiddenWeights[data[idx].types[i]][data[idx].features[i]][j];
            }

            // We flip the piece and side in order to calculate the accumulator from black's perspective
            int piece = data[idx].features[i] / 64;
            int sq = data[idx].features[i] % 64;

            sq ^= 56;
            piece = pieceSwap[piece];

            for(int j = 0; j < neurons; j++) {
                acc[1][j] += hiddenWeights[data[idx].types[i]][piece * 64 + sq][j];
            }
        }

        // Applies relu
        for(int i = 0; i < neurons; i++) {
            acc[0][i] = std::max(0.f, acc[0][i]);
            acc[1][i] = std::max(0.f, acc[1][i]);
        }

        if(data[idx].stm == 0) {
            for(int i = 0; i < neurons; i++) {
                output += acc[0][i] * outputWeights[i];
            }
            for(int i = 0; i < neurons; i++) {
                output += acc[1][i] * outputWeights[neurons + i];
            }
        } else {
            for(int i = 0; i < neurons; i++) {
                output += acc[1][i] * outputWeights[i];
            }
            for(int i = 0; i < neurons; i++) {
                output += acc[0][i] * outputWeights[neurons + i];
            }
        }

        output /= 200;

        float finalOutput = tanh(output);
        mtx.lock();
        loss += (finalOutput - data[idx].target) * (finalOutput - data[idx].target);
        mtx.unlock();

        // Derivative of tanh is sech^2(x)
        // Sech is 1/cosh(x)
        float dLoss = 2 * (finalOutput - data[idx].target) * (1 / cosh(output)) * (1 / cosh(output)) / 200;

        // We store a temporary gradient for the input layer, otherwise the direct accumulation would cause issues
        float hiddenGradient[2][neurons] = {0};

        // Calculates the derivative of output weights
        if(data[idx].stm == 0) {
            for(int i = 0; i < neurons; i++) {
                grad.outputWeightGradient[i] += dLoss * acc[0][i];
                hiddenGradient[0][i] = dLoss * outputWeights[i] * (acc[0][i] > 0 ? 1 : 0);
                grad.hiddenBiasGradient[i] += dLoss * outputWeights[i] * (acc[0][i] > 0 ? 1 : 0);
            }
            for(int i = 0; i < neurons; i++) {
                grad.outputWeightGradient[neurons + i] += dLoss * acc[1][i];
                hiddenGradient[1][i] = dLoss * outputWeights[neurons + i] * (acc[1][i] > 0 ? 1 : 0);
                grad.hiddenBiasGradient[i] += dLoss * outputWeights[neurons + i] * (acc[1][i] > 0 ? 1 : 0);
            }
        } else {
            for(int i = 0; i < neurons; i++) {
                grad.outputWeightGradient[i] += dLoss * acc[1][i];
                hiddenGradient[1][i] = dLoss * outputWeights[i] * (acc[1][i] > 0 ? 1 : 0);
                grad.hiddenBiasGradient[i] += dLoss * outputWeights[i] * (acc[1][i] > 0 ? 1 : 0);
            }
            for(int i = 0; i < neurons; i++) {
                grad.outputWeightGradient[neurons + i] += dLoss * acc[0][i];
                hiddenGradient[0][i] = dLoss * outputWeights[neurons + i] * (acc[0][i] > 0 ? 1 : 0);
                grad.hiddenBiasGradient[i] += dLoss * outputWeights[neurons + i] * (acc[0][i] > 0 ? 1 : 0);
            }
        }

        // Calculate the derivative of each feature
        for(int i = 0; i < 32; i++) {
            if(data[idx].features[i] == 0) break;
            for(int j = 0; j < neurons; j++) {
                grad.hiddenWeightGradient[data[idx].types[i]][data[idx].features[i]][j] += hiddenGradient[0][j];
            }

            int piece = data[idx].features[i] / 64;
            int sq = data[idx].features[i] % 64;

            sq ^= 56;
            piece = pieceSwap[piece];

            for(int j = 0; j < neurons; j++) {
                grad.hiddenWeightGradient[data[idx].types[i]][piece * 64 + sq][j] += hiddenGradient[1][j];
            }
        }
        grad.outputBiasGradient += dLoss;
    }
}

std::pair<NNUEGrad, float> batchGradient(DataLoader &loader) {
    // TODO: Multithread this, after I confirm it to work
    TrainingPosition* data = new TrainingPosition[batchSize];
    loader.fillBatch(data, batchSize, resultPercent);
    NNUEGrad grad;
    float loss = 0;
    int next = 0;
    std::mutex mtx;
    threadedGradCalculator(data, grad, loss, next, mtx);
    delete [] data;
    return {grad, loss / batchSize};
}

void train(DataLoader &loader, int maxBatches, int printInterval) {
    auto start = std::chrono::high_resolution_clock::now();
    ADAM optimizer;
    float runningLoss = -1;

    for(int bC = 0; bC <= maxBatches; bC++) {
        std::pair<NNUEGrad, float> batch = batchGradient(loader);

        optimizer.step(batch.first, lr);

        if(runningLoss == -1) {
            runningLoss = batch.second;
        } else {
            runningLoss *= runningAverageDecay;
            runningLoss += (1 - runningAverageDecay) * batch.second;
        }

        if(bC % printInterval == 0 && bC != 0) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            float u = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            std::cout << "Batch: " << bC << " Loss: " << runningLoss << std::endl;
            std::cout << "Speed: " << (bC * batchSize) / (u/1000.0) << std::endl;
            lr *= lr_decay;
        }
        if(bC % (printInterval * 10) == 0 && bC != 0) {
            for(std::string fen : fens) {
                std::cout << testNNUE(chess::Board(fen, false)) << std::endl;
            }
        }
    }
}