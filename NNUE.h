#pragma once
#include "includes.h"

const int neurons = 16;

float hiddenWeights[18][768][neurons] = {0};
float hiddenBias[neurons] = {0};

float outputWeights[neurons*2] = {0};
float outputBias = 0;

int pieceSwap[12] = {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5};

struct NNUEGrad {
    float hiddenWeightGradient[18][768][neurons] = {0};
    float hiddenBiasGradient[neurons] = {0};

    float outputWeightGradient[neurons*2] = {0};
    float outputBiasGradient = 0;
};

float testNNUE(chess::Board pos) {
    chess::Bitboard occ = pos.all();
    float acc[2][neurons];
    float output = outputBias;

    // Applies the bias
    memcpy(acc[0], hiddenBias, sizeof(float) * neurons);
    memcpy(acc[1], hiddenBias, sizeof(float) * neurons);

    while(occ) {
        int sq = chess::builtin::poplsb(occ);
        int piece = (int)pos.at(sq);
        int type = (int)pos.typeAt(sq);

        // Adds feature to white accumulator
        for(int i = 0; i < neurons; i++) {
            acc[0][i] += hiddenWeights[type][(piece * 64) + sq][i];
        }

        // Flips piece
        piece = pieceSwap[piece];
        sq ^= 56;

        // Adds feature to black accumulator
        for(int i = 0; i < neurons; i++) {
            acc[1][i] += hiddenWeights[type][(piece * 64) + sq][i];
        }
    }

    for(int i = 0; i < neurons; i++) {
        acc[0][i] = std::max(0.f, acc[0][i]);
        acc[1][i] = std::max(0.f, acc[1][i]);
    }
    if(pos.sideToMove() == chess::Color::White) {
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
    
    return output;
}

void writeNNUE(std::string fileName) {
    std::ofstream file(fileName, std::ios::out | std::ios::binary);

    file.write((char*)&hiddenWeights, sizeof(float) * 18 * 768 * neurons);
    file.write((char*)&hiddenBias, sizeof(float) * neurons);

    file.write((char*)&outputWeights, sizeof(float) * 2 * neurons);
    file.write((char*)&outputBias, sizeof(float));
}