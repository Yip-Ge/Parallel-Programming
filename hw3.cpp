#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include "g.h"
#include <deque>


using namespace std;
using namespace chrono;


int main(int argc, char** argv){
    // omp_set_num_threads(9);
    double a = 1;
    double b = 100;
    double eps = 1e-7;
    double s = 12;
    double M = max(g(a), g(b));
    deque<pair<double, double>> ends;
    ends.push_back({a, b});
    deque<pair<double, double>> comb_ends;
    pair<double, double> temp_ends;
    double temp_a, temp_b, ga, gb;
    double loc_M;
    deque<pair<double, double>> temp_comb_ends;
    double thread_num;

   #pragma omp parallel for schedule(dynamic, 1)
        for(int i = 0; i < 1; i++ ){
            printf("Total number of threads: %d\n",omp_get_num_threads());
        }

    auto start = system_clock::now();
    while(!ends.empty()){        
        #pragma omp parallel shared(comb_ends, ends, M, s, eps) private(temp_ends, temp_a, temp_b, ga, gb, loc_M, temp_comb_ends)
        {   
            temp_ends = {0, 0};
            temp_a = 0;
            temp_b = 0;
            ga = 0;
            gb = 0;
            loc_M = M;
            #pragma omp for
                for (auto i=ends.begin(); i < ends.end(); i++){
                    temp_a = i->first;
                    temp_b = i->second;
                    ga = g(temp_a);
                    gb = g(temp_b);
                    loc_M = max(loc_M, max(ga, gb));
                    // cout << "middle M" << M << "\n";
                    if ((ga + gb + s * (temp_b - temp_a))/2 > (loc_M + eps)){
                        temp_comb_ends.push_front({(temp_a+temp_b)/2, temp_b});
                        temp_comb_ends.push_front({temp_a, (temp_a+temp_b)/2});
                    }
                }
            // #pragma omp barrier
            #pragma omp critical
            {
                M = max(M, loc_M);
                while(!temp_comb_ends.empty()){
                    comb_ends.push_back({temp_comb_ends.back().first, temp_comb_ends.back().second});
                    temp_comb_ends.pop_back();
                }
            }
            }
        
        ends.swap(comb_ends);
        while(!comb_ends.empty()){comb_ends.pop_back();}
    }
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "Total time cost:" << double(duration.count()) * microseconds::period::num / microseconds::period::den << "s\n";
    std::cout << "Value of M: " << M << "\n";


}