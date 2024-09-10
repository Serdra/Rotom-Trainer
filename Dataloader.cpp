#include "includes.h"

void DataLoader::fillBatch(TrainingPosition *batch, int batchSize, float resultPercent) {
    for(int i = 0; i < batchSize; i++) {
        for(int j = 0; j < 32; j++) {
            batch[i].features[j] = 0;
        }

        PackedBoard packed;
        infile.read(reinterpret_cast<char*>(&packed), sizeof(packed));

        // Resets file back to start if need be
        if(!infile) {
            infile.clear();
            infile.seekg(0, std::ios::beg);
            infile.read(reinterpret_cast<char*>(&packed), sizeof(packed));
        }

        int idx = 0;
        uint64_t occ = packed.occupied;
        while(occ) {
            int sq = poplsb(occ);
            int piece = packed.pieces[idx];
            int type = piece / 12;
            piece %= 12;

            assert(type < 18);
            assert(piece < 12);
            
            batch[i].features[idx] = (piece * 64) + sq;
            batch[i].types[idx] = type;
            idx++;
        }

        batch[i].target = resultPercent * packed.wdl;
        if(packed.flags & 0b00000001 == 1) {
            batch[i].target *= -1;
            batch[i].stm = 1;
        } else {
            batch[i].stm = 0;
        }
        batch[i].target += (1 - resultPercent) * tanh(packed.eval / 200.f);
    }
}

void DataLoader::shuffle(std::string newFile, uint64_t maxMemory) {
    std::ofstream outFile = std::ofstream(newFile, std::ios::out | std::ios::binary);

    int maxSize = (1024 * 1024 * maxMemory) / sizeof(PackedBoard);
    std::cout << "Shuffling with " << maxSize << " positions per batch" << std::endl;

    std::vector<PackedBoard> vec;
    vec.reserve(maxSize);
    while(true) {
        // Reads an entire batch
        for(int i = 0; i < maxSize; i++) {
            PackedBoard packed;
            infile.read(reinterpret_cast<char*>(&packed), sizeof(packed));
            if(!infile) break;
            vec.push_back(packed);
        }

        // Shuffle vector
        std::random_shuffle(std::begin(vec), std::end(vec));

        // Writes the batch back
        for(int i = 0; i < vec.size(); i++) {
            outFile.write(reinterpret_cast<char*>(&vec[i]), sizeof(vec[i]));
        }

        if(!infile) break;

        vec.clear();
    }
}