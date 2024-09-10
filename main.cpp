#include "includes.h"

int main(int, char**){
    srand(time(0));
    std::default_random_engine generator(19);
    std::normal_distribution<float> distribution(0, 0.6);
    for(int i = 0; i < 18; i++) {
        for(int j = 0; j < 768; j++) {
            for(int k = 0; k < neurons; k++) {
                hiddenWeights[i][j][k] = distribution(generator);
            }
        }
    }
    for(int i = 0; i < neurons*2; i++) {
        outputWeights[i] = distribution(generator);
    }
    for(int i = 0; i < neurons; i++) {
        hiddenBias[i] = distribution(generator);
    }
    outputBias = distribution(generator);

    DataLoader loader("combined_shuf.bin");
    // loader.shuffle("combined_shuf.bin", 8192);
    // return 0;

    train(loader, 120000, 1000);
    writeNNUE("dev.nnue");
    testNNUE(chess::Board("rmnmbm1kmbmnmrm/pmpmpmpmpmpmpmpm/8/8/8/8/PmPmPmPmPmPmPmPm/RmNmBmQmKmBmNmRm w KQkq - 0 1 -", false));
}
