#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <sstream>


template <typename T>
std::vector<T> readCSVToVec(const std::string& filename) {
    std::vector<T> vec;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File [ " << filename << " ] could not be opened\n";
    } else {
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ',')) {
                vec.push_back(static_cast<T>(std::stof(val)));
            }
        }
    }
    file.close();

    return vec;
}

template <typename T>
std::vector<std::vector<T>> readCSVToVecVec(const std::string& filename) {
    std::vector<std::vector<T>> vec;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File [ " << filename << " ] could not be opened\n";
    } else {
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string val;
            std::vector<T> row;
            while (std::getline(ss, val, ',')) {
                row.push_back(static_cast<T>(std::stof(val)));
            }
            vec.push_back(row);
        }
    }
    file.close();

    return vec;
}

template<typename T, uint32_t BatchSize>
bool checkIfBatchTrajsMatch(T* d_xu_traj_batch) {
    std::vector<T> h_xu_traj_batch(TRAJ_SIZE * BatchSize);
    gpuErrchk(cudaMemcpy(h_xu_traj_batch.data(), d_xu_traj_batch, 
        TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

    // Compare each trajectory to the first one
    for (uint32_t i = 1; i < BatchSize; i++) {
        for (uint32_t j = 0; j < TRAJ_SIZE; j++) {
            if (std::abs(h_xu_traj_batch[j] - h_xu_traj_batch[i * TRAJ_SIZE + j]) > 1e-10) {
                std::cout << "Mismatch found at trajectory " << i << ", index " << j << std::endl;
                std::cout << "Expected: " << h_xu_traj_batch[j] 
                         << ", Got: " << h_xu_traj_batch[i * TRAJ_SIZE + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}