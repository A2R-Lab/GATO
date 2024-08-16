#pragma once

#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <sstream>

#include "gato.cuh"

// ---------- Experiment Utils ----------

/**
 * @brief Difference between two timespec structs in microseconds.
 */
#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))


/**
 * @brief Get current timestamp in format YYYYMMDD_HHMMSS.
 */
 std::string getCurrentTimestamp() {
   time_t rawtime;
   struct tm * timeinfo;
   char buffer[80];
   time(&rawtime);
   timeinfo = localtime(&rawtime);
   strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", timeinfo);
   return std::string(buffer);
}


/**
 * @brief Prints properties of all CUDA devices in the system
 */
 void printCudaDeviceProperties() {

   int nDevices;
   cudaGetDeviceCount(&nDevices);

   printf("Number of devices: %d\n", nDevices);

   for (int i = 0; i < nDevices; i++) {
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, i);
   printf("Device Number: %d\n", i);
   printf("  Device name: %s\n", prop.name);
   printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
   printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
   printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
   printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
   printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
   printf("  minor-major: %d-%d\n", prop.minor, prop.major);
   printf("  Warp-size: %d\n", prop.warpSize);
   printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
   printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
   }
}


/**
 * @brief Print current configuration of experiment.
 */
void print_test_config() {
   std::cout << "Knot points: " << gato::KNOT_POINTS << "\n";
   std::cout << "State size: " << gato::STATE_SIZE << "\n";
   std::cout << "Datatype: " << (USE_DOUBLES ? "DOUBLE" : "FLOAT") << "\n";
   std::cout << "Sqp exits condition: " << (CONST_UPDATE_FREQ ? "CONSTANT TIME" : "CONSTANT ITERS") << "\n";
   std::cout << "QD COST: " << QD_COST << "\n";
   std::cout << "R COST: " << R_COST << "\n";
   std::cout << "Rho factor: " << RHO_FACTOR << "\n";
   std::cout << "Rho max: " << RHO_MAX << "\n";
   std::cout << "Test iters: " << TEST_ITERS << "\n";
#if CONST_UPDATE_FREQ
   std::cout << "Max sqp time: " << SQP_MAX_TIME_US << "\n";
#else
   std::cout << "Max sqp iter: " << SQP_MAX_ITER << "\n";
#endif
   std::cout << "Solver: " << ((LINSYS_SOLVE == 1) ? "PCG" : "QDLDL") << "\n";
#if LINSYS_SOLVE == 1
   std::cout << "Max pcg iter: " << PCG_MAX_ITER << "\n";
#endif
   std::cout << "Save data: " << (SAVE_DATA ? "ON" : "OFF") << "\n";
   std::cout << "Jitters: " << (REMOVE_JITTERS ? "ON" : "OFF") << "\n\n\n";
}


/**
 * @brief Print statistics for a vector.
 * @param data Vector of data to print statistics for.
 */
template<bool PRINT_DISTRIBUTION = true>
void printStats(std::vector<double> *data) {
   double sum = std::accumulate(data->begin(), data->end(), 0.0);
   double mean = sum/static_cast<double>(data->size());
   std::vector<double> diff(data->size());
   std::transform(data->begin(), data->end(), diff.begin(), [mean](double x) {return x - mean;});
   double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
   double stdev = std::sqrt(sq_sum / data->size());
   std::vector<double>::iterator minInd = std::min_element(data->begin(), data->end());
   std::vector<double>::iterator maxInd = std::max_element(data->begin(), data->end());
   double min = data->at(std::distance(data->begin(), minInd)); 
   double max = data->at(std::distance(data->begin(), maxInd));
   printf("Average[%fus] Std Dev [%fus] Min [%fus] Max [%fus] \n",mean,stdev,min,max);
   if (PRINT_DISTRIBUTION){
      double hist[] = {0,0,0,0,0,0,0};
      for(int i = 0; i < data->size(); i++){
         double value = data->at(i);
         if (value < mean - stdev){
            if (value < mean - 2*stdev){
               if (value < mean - 3*stdev){hist[0] += 1.0;}
               else{hist[1] += 1.0;}
            }
            else{hist[2] += 1.0;}
         }
         else if (value > mean + stdev){
            if (value > mean + 2*stdev){
               if (value > mean + 3*stdev){hist[6] += 1.0;}
               else{hist[5] += 1.0;}
            }
            else{hist[4] += 1.0;}
         }
         else{hist[3] += 1.0;}
      }
      for(int i = 0; i < 7; i++){hist[i] = (hist[i]/static_cast<double>(data->size()))*100;}
      printf("    Distribution |  -3  |  -2  |  -1  |   0  |   1  |   2  |   3  |\n");
      printf("    (X std dev)  | %2.2f | %2.2f | %2.2f | %2.2f | %2.2f | %2.2f | %2.2f |\n",
                                hist[0],hist[1],hist[2],hist[3],hist[4],hist[5],hist[6]);
      std::sort(data->begin(), data->end()); 
      printf("    Percentiles |  50   |  60   |  70   |  75   |  80   |  85   |  90   |  95   |  99   |\n");
      printf("                | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n",
                              data->at(data->size()/2),data->at(data->size()/5*3),data->at(data->size()/10*7),
                              data->at(data->size()/4*3),data->at(data->size()/5*4),data->at(data->size()/20*17),
                              data->at(data->size()/10*9),data->at(data->size()/20*19),data->at(data->size()/100*99));
      bool onePer = false; bool twoPer = false; bool fivePer = false; bool tenPer = false;
      for(int i = 0; i < data->size(); i++){
         if(!onePer && data->at(i) >= mean * 1.01){ onePer = true;
            printf("    More than 1 Percent above mean at [%2.2f] Percentile\n",static_cast<double>(i)/static_cast<double>(data->size())*100.0);
         }
         if(!twoPer && data->at(i) >= mean * 1.02){ twoPer = true;
            printf("    More than 2 Percent above mean at [%2.2f] Percentile\n",static_cast<double>(i)/static_cast<double>(data->size())*100.0);
         }
         if(!fivePer && data->at(i) >= mean * 1.05){ fivePer = true;
            printf("    More than 5 Percent above mean at [%2.2f] Percentile\n",static_cast<double>(i)/static_cast<double>(data->size())*100.0);
         }
         if(!tenPer && data->at(i) >= mean * 1.10){ tenPer = true;
            printf("    More than 10 Percent above mean at [%2.2f] Percentile\n",static_cast<double>(i)/static_cast<double>(data->size())*100.0);
         }
      }
   }
}

/**
 * @brief Print formatted string with statistics for a vector of data.
 * @param data  Vector of data
 * @param prefix
 * @return Formatted string with statistics.
 */
template<typename T>
std::string printStats(std::vector<T> *data, std::string prefix = "data") {
   T sum = std::accumulate(data->begin(), data->end(), static_cast<T>(0));
   float mean = sum/static_cast<double>(data->size());
   std::vector<T> diff(data->size());
   std::transform(data->begin(), data->end(), diff.begin(), [mean](T x) {return x - mean;});
   T sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
   T stdev = std::sqrt(sq_sum / data->size());
   typename std::vector<T>::iterator minInd = std::min_element(data->begin(), data->end());
   typename std::vector<T>::iterator maxInd = std::max_element(data->begin(), data->end());
   T min = data->at(std::distance(data->begin(), minInd)); 
   T max = data->at(std::distance(data->begin(), maxInd));

   // Now also want to sort and get median, first and third quartile for variance plot
   std::vector<T> sortedData(*data);
   std::sort(sortedData.begin(), sortedData.end());

   std::cout << std::endl;
   T median, Q1, Q3;
   size_t n = sortedData.size();
   if (n % 2 == 0) {
      median = (sortedData[n/2 - 1] + sortedData[n/2]) / 2.0;
      Q1 = (sortedData[n/4 - 1] + sortedData[n/4]) / 2.0;
      Q3 = (sortedData[3*n/4 - 1] + sortedData[3*n/4]) / 2.0;
   } else {
      median = sortedData[n/2];
      Q1 = sortedData[n/4];
      Q3 = sortedData[3*n/4];
   }
   std::cout << "Average [" << mean << "] Std Dev [" << stdev << "] Min [" << min << "] Max [" << max << "] Median [" << median << "] Q1 [" << Q1 << "] Q3 [" << Q3 << "]" << std::endl;

   // Construct the formatted string
   std::stringstream ss;
   ss << "Average [" << mean << "] Std Dev [" << stdev << "] Min [" << min << "] Max [" << max << "] Median [" << median << "] Q1 [" << Q1 << "] Q3 [" << Q3 << "]";
   
   return ss.str();
}


