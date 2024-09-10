#pragma once
#include "includes.h"

int poplsb(uint64_t &b) {
    auto index = int(__builtin_ctzll(b));
    b &= b - 1;
    return index;
}

struct TrainingPosition {
    // Features from white perspective
    uint16_t features[32] = {0};
    uint8_t types[32] = {0};

    // Target from STM perspective probably idk yet
    float target = 0;

    // 1 if black
    int stm = 0;
};

// TODO: Add more flags or condense wdl and flags
struct PackedBoard {
    // Pieces are not stm relative
    chess::Bitboard occupied = 0;

    // piece / 12 = pokemon type
    // piece % 12 = piece type
    uint8_t pieces[32] = {0}; 

    // Eval is not stm relative
    int16_t eval = 0;

    // 0 if draw, 1 if white win, -1 if black win
    int8_t wdl = 0;

    // 0b00000001 stm, 0 if white 1 if black
    // 0b00000010 in check
    uint8_t flags = 0; 
};

class DataLoader {
   private:
    std::ifstream infile;

   public:
    DataLoader(std::string name) {
        infile = std::ifstream(name, std::ios::in |  std::ios::binary);
        std::cout << infile.is_open() << std::endl;
    }

    void fillBatch(TrainingPosition *batch, int batchSize, float resultPercent);

    void shuffle(std::string newFile, uint64_t maxMemory);
};

std::string fens[] = {
    "rmnmbmqmkmbmnmrm/pmpmpmpmpmpmpmpm/8/8/8/8/PmPmPmPmPmPmPmPm/RmNmBmQmKmBmNmRm w KQkq - 0 1 -", // Startpos, should be slightly above 0
    "rmnmbm1kmbmnmrm/pmpmpmpmpmpmpmpm/8/8/8/8/PmPmPmPmPmPmPmPm/RmNmBmQmKmBmNmRm w KQkq - 0 1 -" // High material advantage
};