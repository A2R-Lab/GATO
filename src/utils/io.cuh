#pragma once

#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>


// ---------- IO Utils ----------

// parameters for logging for MPCGPU experiments
struct MPCLogParams {
    int start_state_index;
    int goal_state_index;
    uint32_t test_iter;
    std::string test_output_prefix;
};

template <typename T>
void dump_tracking_data(
    const std::vector<int>& pcg_iters, 
    const std::vector<bool>& pcg_exits, 
    const std::vector<double>& linsys_times, 
    const std::vector<double>& sqp_times, 
    const std::vector<uint32_t>& sqp_iters, 
    const std::vector<bool>& sqp_exits, 
    const std::vector<T>& tracking_errors, 
    const std::vector<std::vector<T>>& tracking_path, 
    uint32_t timesteps_taken, 
    uint32_t control_updates_taken, 
    const MPCLogParams& mpc_log_params) 
{
    auto createFileName = [&](const std::string& data_type) {
        return mpc_log_params.test_output_prefix + "_" + std::to_string(mpc_log_params.test_iter) + "_" + data_type + ".result";
    };
    
    // Helper function to dump single-dimension vector data
    auto dumpVectorData = [&](const auto& data, const std::string& data_type) {
        std::ofstream file(createFileName(data_type));
        if (!file.is_open()) {
            std::cerr << "Failed to open " << data_type << " file.\n";
            return;
        }
        for (const auto& item : *data) {
            file << item << '\n';
        }
        file.close();
    };

    // Dump single-dimension vector data
    dumpVectorData(pcg_iters, "pcg_iters");
    dumpVectorData(linsys_times, "linsys_times");
    dumpVectorData(sqp_times, "sqp_times");
    dumpVectorData(sqp_iters, "sqp_iters");
    dumpVectorData(sqp_exits, "sqp_exits");
    dumpVectorData(tracking_errors, "tracking_errors");
    dumpVectorData(pcg_exits, "pcg_exits");

    // Dump two-dimension vector data (tracking_path)
    std::ofstream file(createFileName("tracking_path"));
    if (!file.is_open()) {
        std::cerr << "Failed to open tracking_path file.\n";
        return;
    }
    for (const auto& outerItem : *tracking_path) {
        for (const auto& innerItem : outerItem) {
            file << innerItem << ',';
        }
        file << '\n';
    }
    file.close();

    std::ofstream statsfile(createFileName("stats"));
    if (!statsfile.is_open()) {
        std::cerr << "Failed to open stats file.\n";
        return;
    }
    statsfile << "timesteps: " << timesteps_taken << "\n";
    statsfile << "control_updates: " << control_updates_taken << "\n";
    statsfile.close();
}

// read a CSV file into a vector of vectors
template <typename T>
std::vector<std::vector<T>> readCSVToVecVec(const std::string& filename) {

    std::vector<std::vector<T>> data;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "File [ " << filename << " ] could not be opened!\n";
    } else {
        std::string line;
        while (std::getline(infile, line)) {
            std::vector<T> row;
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ',')) {
                row.push_back(std::stof(val));
            }
            data.push_back(row);
        }
    }
    infile.close();
    
    return data;
}

// read a CSV file into a vector
template <typename T>
std::vector<T> readCSVToVec(const std::string& filename) {

    std::vector<T> data;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "File [ " << filename << " ] could not be opened!\n";
    } else {
        std::string line;
        while (std::getline(infile, line)) {
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ',')) {
                data.push_back(static_cast<T>(std::stof(val)));
            }
        }
    }
    infile.close();

    return data;
}

// Format stats string values into CSV format
std::string getStatsString(const std::string& statsString) {

    std::stringstream ss(statsString);
    std::string token;
    std::string csvFormattedString;
    while (getline(ss, token, '[')) {
        if (getline(ss, token, ']')) {
            if (!csvFormattedString.empty()) {
                csvFormattedString += ",";
            }
            csvFormattedString += token;
        }
    }
    
    return csvFormattedString;
 }

 // Write experiment results to a CSV file
int writeResultsToCSV(const std::string& filename, const std::string& trackingStats, const std::string& linsysOrSqpStats){

   std::ofstream csvFile(filename);
   if (!csvFile.is_open()) {
       std::cerr << "Error opening CSV file for writing." << std::endl;
       return 1;
   }

   // Write to file
   csvFile << "Average, Std Dev, Min, Max, Median, Q1, Q3\n";
   csvFile << getStatsString(trackingStats) << "\n";
   csvFile << getStatsString(linsysOrSqpStats) << "\n";

   csvFile.close();

   return 0;
}