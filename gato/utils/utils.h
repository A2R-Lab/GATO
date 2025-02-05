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