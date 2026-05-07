/*
 * SIFT OpenCL Kernel Profiler
 * 
 * Profiles individual SIFT kernels to identify performance bottlenecks
 * and optimization opportunities.
 * 
 * Usage: g++ -std=c++11 sift_kernel_profiler.cpp -o sift_kernel_profiler \
 *           `pkg-config --cflags --libs opencv4`
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace cv;

struct KernelProfile {
    string name;
    int count;
    double total_ms;
    double min_ms;
    double max_ms;
    double avg_ms;
    
    double utilization_percent;  // GPU utilization during this kernel (if available)
    
    KernelProfile(const string& n) 
        : name(n), count(0), total_ms(0), min_ms(1e9), max_ms(0), 
          avg_ms(0), utilization_percent(0.0) {}
};

class SiftProfiler {
public:
    static map<string, KernelProfile> profiles;
    
    static void resetProfiles() {
        profiles.clear();
    }
    
    static void recordKernel(const string& name, double ms) {
        if (profiles.find(name) == profiles.end()) {
            profiles[name] = KernelProfile(name);
        }
        
        KernelProfile& prof = profiles[name];
        prof.count++;
        prof.total_ms += ms;
        prof.min_ms = min(prof.min_ms, ms);
        prof.max_ms = max(prof.max_ms, ms);
        prof.avg_ms = prof.total_ms / prof.count;
    }
    
    static void printReport(const string& title = "SIFT Kernel Profiling Report") {
        cout << "\n" << string(100, '=') << "\n";
        cout << title << "\n";
        cout << string(100, '=') << "\n\n";
        
        cout << left 
             << setw(35) << "Kernel Name"
             << setw(12) << "Count"
             << setw(12) << "Total (ms)"
             << setw(12) << "Avg (ms)"
             << setw(12) << "Min (ms)"
             << setw(12) << "Max (ms)"
             << setw(12) << "% of Total"
             << "\n";
        cout << string(100, '-') << "\n";
        
        double total = 0;
        for (auto& p : profiles) {
            total += p.second.total_ms;
        }
        
        vector<pair<string, KernelProfile*>> sorted;
        for (auto& p : profiles) {
            sorted.push_back({p.first, &p.second});
        }
        sort(sorted.begin(), sorted.end(),
             [](const auto& a, const auto& b) {
                 return a.second->total_ms > b.second->total_ms;
             });
        
        for (auto& item : sorted) {
            const KernelProfile& prof = *item.second;
            double percent = (total > 0) ? (prof.total_ms / total * 100) : 0;
            
            cout << left
                 << setw(35) << prof.name
                 << setw(12) << prof.count
                 << setw(12) << fixed << setprecision(3) << prof.total_ms
                 << setw(12) << fixed << setprecision(3) << prof.avg_ms
                 << setw(12) << fixed << setprecision(3) << prof.min_ms
                 << setw(12) << fixed << setprecision(3) << prof.max_ms
                 << setw(12) << fixed << setprecision(1) << percent
                 << "%\n";
        }
        
        cout << string(100, '-') << "\n";
        cout << "Total GPU kernel time: " << fixed << setprecision(3) << total << " ms\n";
        cout << string(100, '=') << "\n\n";
    }
};

map<string, KernelProfile> SiftProfiler::profiles;

// Helper function to measure kernel execution
template<typename Func>
double measureKernel(const string& name, Func&& kernel_fn, int iterations = 1) {
    cv::ocl::finish();  // Ensure GPU is idle before timing
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        kernel_fn();
    }
    
    cv::ocl::finish();  // Wait for completion
    
    auto end = chrono::high_resolution_clock::now();
    double elapsed_ms = chrono::duration<double, milli>(end - start).count();
    
    double avg_ms = elapsed_ms / iterations;
    SiftProfiler::recordKernel(name, avg_ms);
    
    return avg_ms;
}

int main(int argc, char* argv[]) {
    cout << "OpenCV SIFT Kernel Profiler\n";
    cout << "============================\n\n";
    
    // Check OpenCL availability
    if (!cv::ocl::haveOpenCL()) {
        cerr << "ERROR: OpenCL not available\n";
        return 1;
    }
    
    cout << "OpenCL Device: " << cv::ocl::Device::getDefault().name() << "\n";
    cout << "OpenCL Version: " << cv::ocl::Device::getDefault().version() << "\n\n";
    
    // Enable OpenCL
    cv::ocl::setUseOpenCL(true);
    
    // Create test images and parameters
    vector<int> image_sizes = {384, 720, 1080, 2160};
    vector<string> size_names = {"S_small", "M_medium", "L_large", "XL_very_large"};
    
    map<string, vector<double>> kernel_times;
    
    for (size_t i = 0; i < image_sizes.size(); ++i) {
        int size = image_sizes[i];
        int width = size * 4 / 3;   // 4:3 aspect ratio
        int height = size;
        
        cout << "Profiling " << size_names[i] << " (" << width << "x" << height << ")...\n";
        
        // Create test image
        Mat img(height, width, CV_8UC1);
        randu(img, 0, 256);
        
        UMat uimg = img.getUMat(ACCESS_READ);
        UMat descriptors;
        vector<KeyPoint> keypoints;
        
        // Use SIFT with detectAndCompute
        Ptr<cv::SIFT> sift = cv::SIFT::create();
        
        SiftProfiler::resetProfiles();
        
        // Warmup
        sift->detectAndCompute(uimg, UMat(), keypoints, descriptors);
        
        // Timed runs
        for (int iter = 0; iter < 3; ++iter) {
            keypoints.clear();
            descriptors.release();
            sift->detectAndCompute(uimg, UMat(), keypoints, descriptors);
        }
        
        cout << "  Keypoints detected: " << keypoints.size() << "\n";
        cout << "  Descriptor shape: " << descriptors.size() << "\n";
        
        SiftProfiler::printReport("Kernels for " + size_names[i]);
        
        // Store metrics for summary
        for (auto& p : SiftProfiler::profiles) {
            kernel_times[p.first].push_back(p.second.avg_ms);
        }
    }
    
    // Print summary across all sizes
    cout << "\n" << string(100, '=') << "\n";
    cout << "KERNEL SCALING ANALYSIS (Time vs Image Size)\n";
    cout << string(100, '=') << "\n\n";
    
    cout << left
         << setw(35) << "Kernel Name"
         << setw(15) << "S_small"
         << setw(15) << "M_medium"
         << setw(15) << "L_large"
         << setw(15) << "XL_very_large"
         << setw(15) << "Scaling Factor"
         << "\n";
    cout << string(100, '-') << "\n";
    
    for (auto& kv : kernel_times) {
        const vector<double>& times = kv.second;
        if (times.size() >= 4) {
            double scale = times[3] / times[0];  // XL / S_small ratio
            cout << left
                 << setw(35) << kv.first
                 << setw(15) << fixed << setprecision(3) << times[0]
                 << setw(15) << fixed << setprecision(3) << times[1]
                 << setw(15) << fixed << setprecision(3) << times[2]
                 << setw(15) << fixed << setprecision(3) << times[3]
                 << setw(15) << fixed << setprecision(1) << scale << "x\n";
        }
    }
    
    cout << "\n" << string(100, '=') << "\n";
    cout << "INTERPRETATION GUIDE:\n";
    cout << "  • Linear scaling (4x time for 4x pixels) = Memory-bound\n";
    cout << "  • Sub-linear scaling (2x time for 4x pixels) = Compute-bound\n";
    cout << "  • Super-linear scaling (10x time for 4x pixels) = Occupancy/dispatch overhead\n";
    cout << string(100, '=') << "\n\n";
    
    return 0;
}
