// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
//
// Copyright (c) 2017 - 2024 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

// Config files
#include <SAMRAI_config.h>

// Headers for basic PETSc functions
#include <petscsys.h>

// Headers for basic SAMRAI objects
#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

// Headers for application-specific algorithm/data structure objects
#include <ibamr/ConstraintIBMethod.h>
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBHydrodynamicForceEvaluator.h>
#include <ibamr/IBStandardForceGen.h>
#include <ibamr/IBStandardInitializer.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/INSStaggeredPressureBcCoef.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/LData.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

// Set up application namespace declarations
#include <ibamr/app_namespaces.h>

// Application objects
#include "IBEELKinematics.h"
#include "IBCylinderKinematics.h"

// Python interface headers
#include <Python.h>

// Add mutex for thread safety
#include <mutex>

// Global variable to store current amplitude
double current_amplitude = 1.0;

// Global Python objects
PyObject *pModule = NULL, *pFunc = NULL;
bool python_initialized = false;

// Static mutex for kinematics operations
static std::mutex kinematics_mutex;

// JSON library header
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Socket-related headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>  // for close()

// Timing-related headers
#include <chrono>

// Additional necessary headers
#include <fcntl.h>
#include <errno.h>

// Standard library headers
#include <algorithm>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <queue>

// Data structure definitions
struct FishPressureData {
    std::vector<double> pressures;
    std::vector<std::vector<double>> coordinates;
};

// Structure for caching state records
struct StateRecord {
    double time;
    std::vector<FishPressureData> fish_data;
    std::vector<std::vector<double>> inertia_power;
    std::vector<std::vector<double>> constraint_power;
    std::vector<std::vector<double>> velocities;
    std::vector<std::vector<double>> angular_velocities;
};

// Python initialization and cleanup function declarations
void initialize_python() {
    if (!python_initialized) {
        // Set Python program name to point to Anaconda's Python interpreter
        wchar_t program[100];
        mbstowcs(program, "/home/u/anaconda3/bin/python", 100);
        Py_SetProgramName(program);
        
        Py_Initialize();
        
        // Add Anaconda path to Python search path
        PyRun_SimpleString("import sys\n"
                         "sys.path.insert(0, '/home/u/anaconda3/lib/python3.12/site-packages')\n"
                         "sys.path.append('.')\n");
        
        PyObject *pName = PyUnicode_FromString("fish_controller");
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);
        
        if (pModule != NULL) {
            pFunc = PyObject_GetAttrString(pModule, "calculate_amplitude");
            pout << "Successfully loaded fish_controller module\n";
        } else {
            pout << "Warning: Unable to load fish_controller module, will use default amplitude value\n";
            PyErr_Print(); // Print detailed Python error information
        }
        
        python_initialized = true;
    }
}

void finalize_python() {
    if (python_initialized) {
        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
        Py_Finalize();
        python_initialized = false;
    }
}

// Return structure definition
struct ControllerResult {
    std::vector<double> amplitudes;
    std::vector<double> T_n_values;
};

// Modified call_python_controllers_batch function
ControllerResult call_python_controllers_batch(
    const std::vector<FishPressureData>& fish_data, 
    double time,
    Pointer<ConstraintIBMethod> ib_method_ops,
    int server_port) { // Add port parameter
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    pout << "Starting Python controller call...\n";
    
    // Default return values
    ControllerResult result;
    result.amplitudes.resize(fish_data.size(), current_amplitude);
    result.T_n_values.resize(fish_data.size(), 1.0); // Default T_n value
    
    // Only execute communication on main process
    const int mpi_rank = IBTK_MPI::getRank();
    if (mpi_rank != 0) {
        return result;
    }
    
    int sockfd = -1;
    
    try {
        // Create socket
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            pout << "Error: Unable to create socket\n";
            return result;
        }
        
        // Set server address
        struct sockaddr_in serv_addr;
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(server_port); // Use passed port
        const char* head_node_ip = std::getenv("HEAD_NODE_IP");
        if (head_node_ip == nullptr) {
            pout << "Warning: HEAD_NODE_IP environment variable not set, will use 127.0.0.1\n";
            head_node_ip = "127.0.0.1";
        }
        serv_addr.sin_addr.s_addr = inet_addr(head_node_ip);
        
        // Connect to server
        if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            pout << "Error: Unable to connect to server\n";
            close(sockfd);
            return result;
        }
        
        // Create JSON for communication
        json request;
        request["time"] = time;

        // Add environment ID - obtained from current working directory
        char* cwd = getcwd(NULL, 0);
        std::string current_dir(cwd);
        free(cwd);
        size_t pos = current_dir.find_last_of('/');
        if (pos != std::string::npos) {
            request["env_id"] = current_dir.substr(pos + 1);  // e.g., "env1", "env2"
        } else {
            request["env_id"] = "unknown_env";
        }

        request["request_type"] = "control";
        request["fish_data"] = json::array();

        // Add fish data
        for (const auto& fish : fish_data) {
            json fish_json;
            fish_json["pressures"] = fish.pressures;
            fish_json["coordinates"] = fish.coordinates;
            request["fish_data"].push_back(fish_json);
        }
        
        // Add velocity data
        const std::vector<std::vector<double>>& velocities = ib_method_ops->getRigidTranslationalVelocity();
        request["velocities"] = json::array();
        for (const auto& vel : velocities) {
            json vel_json = {
                {"x", vel[0]},
                {"y", vel[1]}
            };
            request["velocities"].push_back(vel_json);
        }
        
        // Add angular velocity data
        const std::vector<std::vector<double>>& rot_velocities = ib_method_ops->getStructureRotationalMomentum();
        request["angular_velocities"] = json::array();
        if (!rot_velocities.empty()) {
            json rot_vel_json = {
                {"z", rot_velocities[0][2]}  // Only take fish's angular velocity
            };
            request["angular_velocities"].push_back(rot_vel_json);
        }
        
        // Add power data
        const std::vector<std::vector<double>>& inertia_power = ib_method_ops->getInertiaPower();
        const std::vector<std::vector<double>>& constraint_power = ib_method_ops->getConstraintPower();
        request["power_data"] = json::array();
        if (!inertia_power.empty() && !constraint_power.empty()) {
            json power_json;
            power_json["inertia"] = {
                {"x", inertia_power[0][0]},  // Only take fish's inertia power
                {"y", inertia_power[0][1]}
            };
            power_json["constraint"] = {
                {"x", constraint_power[0][0]},  // Only take fish's constraint power
                {"y", constraint_power[0][1]}
            };
            request["power_data"].push_back(power_json);
        }
        
        
        // Send request
        std::string json_str = request.dump() + "\n";  // Add newline as message end marker
        send(sockfd, json_str.c_str(), json_str.length(), 0);
        
        // Receive response
        char buffer[4096] = {0};
        std::string response;
        
        int n = recv(sockfd, buffer, sizeof(buffer) - 1, 0);
        if (n > 0) {
            buffer[n] = '\0';
            response = buffer;
        }
        
        // Close socket
        close(sockfd);
        sockfd = -1;
        
        // Parse response
        if (!response.empty()) {
            try {
                // Remove possible trailing newline
                if (response.back() == '\n') {
                    response.pop_back();
                }
                
                json response_json = json::parse(response);
                if (response_json.contains("amplitudes")) {
                    std::vector<double> amplitudes = response_json["amplitudes"].get<std::vector<double>>();
                    
                    // Validate returned data
                    if (amplitudes.size() == fish_data.size()) {
                        result.amplitudes = amplitudes;
                    }
                }
                
                // Parse T_n values
                if (response_json.contains("T_n_values")) {
                    std::vector<double> T_n_values = response_json["T_n_values"].get<std::vector<double>>();
                    
                    // Validate returned data
                    if (T_n_values.size() == fish_data.size()) {
                        result.T_n_values = T_n_values;
                    }
                }
            } catch (...) {
                // Ignore errors
            }
        }
        
    } catch (...) {
        // Ignore errors
    }
    
    // Ensure socket is closed
    if (sockfd >= 0) {
        close(sockfd);
    }
    
    // Calculate time consumption
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    pout << "Python controller call completed, time consumed: " << duration.count() << " milliseconds\n";
    
    return result;
}

// Function declarations
std::vector<FishPressureData> collect_fish_pressure_data(
    Pointer<PatchHierarchy<NDIM>> patch_hierarchy,
    Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
    LDataManager* l_data_manager,
    const int p_idx,
    const int num_structures,
    Pointer<ConstraintIBMethod> ib_method_ops);

void output_data(Pointer<PatchHierarchy<NDIM> > patch_hierarchy,
                 Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
                 LDataManager* l_data_manager,
                 const int iteration_num,
                 const double loop_time,
                 const string& data_dump_dirname);

// Function declaration for batch sending state records
void send_state_record_batch(
    const std::vector<StateRecord>& record_cache,
    int server_port);

// Function declaration for clustering fish by coordinates
std::vector<std::vector<int>> clusterFishByCoordinates(
    Pointer<PatchHierarchy<NDIM>> patch_hierarchy,
    LDataManager* l_data_manager);

// Function declarations for processing head contours
int processHeadContour(const std::vector<std::pair<int, std::vector<double>>>& head_contour_points,
                       std::vector<std::pair<int, std::vector<double>>>& sampled_points,
                       size_t struct_id, int mpi_rank,
                       const std::vector<std::pair<int, std::vector<double>>>& fish_points,
                       bool head_at_larger_coord,
                       bool use_fallback_strategy);

int processHeadContourVertical(const std::vector<std::pair<int, std::vector<double>>>& head_contour_points,
                               std::vector<std::pair<int, std::vector<double>>>& sampled_points,
                               size_t struct_id, int mpi_rank,
                               const std::vector<std::pair<int, std::vector<double>>>& fish_points,
                               bool head_at_larger_coord,
                               bool use_fallback_strategy);

// Simplified version that only collects data on root process
// Modified collect_fish_pressure_data function to get pressure for specific IDs from global data
std::vector<FishPressureData> collect_fish_pressure_data(
    Pointer<PatchHierarchy<NDIM>> patch_hierarchy,
    Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
    LDataManager* l_data_manager,
    const int p_idx,
    const int num_structures,
    Pointer<ConstraintIBMethod> ib_method_ops) {
    
    const int mpi_rank = IBTK_MPI::getRank();
    const int mpi_size = IBTK_MPI::getNodes();
    
    // Get fish body point index sets through clustering analysis (only returns fish body)
    std::vector<std::vector<int>> fish_clusters = clusterFishByCoordinates(
        patch_hierarchy, l_data_manager);

    // Check if fish body was found
    if (fish_clusters.empty()) {
        if (mpi_rank == 0) {
            pout << "Warning: No fish body clusters found, skipping pressure data collection" << std::endl;
        }
        // Return empty fish data
        std::vector<FishPressureData> empty_data;
        return empty_data;
    }
    
    if (mpi_rank == 0) {
        pout << "Found " << fish_clusters.size() << " fish, starting pressure data collection" << std::endl;
    }
    
    // Create return data structure
    std::vector<FishPressureData> fish_data(fish_clusters.size());
    
    // Get finest hierarchy level
    const int finest_hier_level = patch_hierarchy->getFinestLevelNumber();
    
    // Specify point ID list for pressure acquisition
    std::vector<int> pressure_point_ids = {0, 100, 200};

    
    // Integrate all interested point coordinates on root process
    std::vector<std::vector<double>> all_target_coords;
    std::vector<int> target_fish_ids;
    std::vector<int> target_point_ids;
    
    // Get local Lagrangian point data
    Pointer<LData> X_data = l_data_manager->getLData("X", finest_hier_level);
    const int local_node_count = X_data->getLocalNodeCount();
    double* X_data_array = X_data->getLocalFormVecArray()->data();


    // Collect how many points each process has
    std::vector<int> all_counts(mpi_size, 0);
    MPI_Allgather(&local_node_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);


    // Calculate offsets
    std::vector<int> displacements(mpi_size, 0);
    int total_count = 0;
    for (int i = 0; i < mpi_size; ++i) {
        displacements[i] = total_count;
        total_count += all_counts[i];
    }

    // Create local coordinate array - all processes prepare
    std::vector<double> local_coords(local_node_count * NDIM);
    for (int i = 0; i < local_node_count; ++i) {
        for (int d = 0; d < NDIM; ++d) {
            local_coords[i * NDIM + d] = X_data_array[i * NDIM + d];
        }
    }

    // Only root process allocates receive buffer
    std::vector<double> all_lagrangian_coords;
    if (mpi_rank == 0) {
        all_lagrangian_coords.resize(total_count * NDIM);
    }

    // Prepare MPI_Gatherv parameters - all processes prepare
    std::vector<int> coord_counts(mpi_size);
    std::vector<int> coord_displacements(mpi_size);
    for (int i = 0; i < mpi_size; ++i) {
        coord_counts[i] = all_counts[i] * NDIM;
        coord_displacements[i] = displacements[i] * NDIM;
    }

    // All processes execute MPI_Gatherv
    MPI_Gatherv(local_coords.data(), local_node_count * NDIM, MPI_DOUBLE,
        all_lagrangian_coords.data(), coord_counts.data(), coord_displacements.data(),
        MPI_DOUBLE, 0, PETSC_COMM_WORLD);

    // Release data array
    X_data->restoreArrays();



if (mpi_rank == 0) {
    // Select edge points for each fish and perform sampling
    for (size_t struct_id = 0; struct_id < fish_clusters.size(); ++struct_id) {
        const std::vector<int>& cluster_indices = fish_clusters[struct_id];
        std::vector<std::pair<int, std::vector<double>>> sampled_points;
        
        // Collect all edge points of this fish
        std::vector<std::pair<int, std::vector<double>>> fish_points;
        double min_x = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double min_y = std::numeric_limits<double>::max();
        double max_y = std::numeric_limits<double>::lowest();

        // 1. First collect all points and their coordinates, while recording x and y ranges
        for (int global_idx : cluster_indices) {
            std::vector<double> point_coord(NDIM);
            for (int d = 0; d < NDIM; ++d) {
                point_coord[d] = all_lagrangian_coords[global_idx * NDIM + d];
            }
            
            fish_points.push_back({global_idx, point_coord});
            min_x = std::min(min_x, point_coord[0]);
            max_x = std::max(max_x, point_coord[0]);
            min_y = std::min(min_y, point_coord[1]);
            max_y = std::max(max_y, point_coord[1]);
        }
        
        // 2. Determine fish swimming direction and head position
        double x_span = max_x - min_x;
        double y_span = max_y - min_y;
        
        bool is_horizontal = (x_span > y_span);  // true=swimming left-right, false=swimming up-down
        
        // Determine head position based on velocity direction
        bool head_at_larger_coord = false; // Default value
        const std::vector<std::vector<double>>& velocities = ib_method_ops->getRigidTranslationalVelocity();

        if (mpi_rank == 0 && !velocities.empty() && struct_id < velocities.size()) {
             const std::vector<double>& vel = velocities[struct_id];
             double vx = vel[0];
             double vy = vel[1];

             if (is_horizontal) {
                 if (std::abs(vx) > 1e-6) {
                     head_at_larger_coord = (vx > 0); // vx > 0 -> head on right(large), vx < 0 -> head on left(small)
                 }
                 pout << "[Process " << mpi_rank << "] Fish " << struct_id 
                      << " horizontal swimming, vx=" << vx << ". Head determined to be on " 
                      << (head_at_larger_coord ? "right side (x large)" : "left side (x small)") << ".\n";
             } else {
                 if (std::abs(vy) > 1e-6) {
                     head_at_larger_coord = (vy > 0); // vy > 0 -> head on top(large), vy < 0 -> head on bottom(small)
                 }
                 pout << "[Process " << mpi_rank << "] Fish " << struct_id 
                      << " vertical swimming, vy=" << vy << ". Head determined to be on " 
                      << (head_at_larger_coord ? "top side (y large)" : "bottom side (y small)") << ".\n";
             }
        } else if (mpi_rank == 0) {
             pout << "[Process " << mpi_rank << "] Fish " << struct_id << " unable to get velocity, using default head direction (smaller coordinate end).\n";
        }
        
        // 3. Adjust head detection algorithm based on determination result (simplified with error correction logic)
        if (is_horizontal) {
            std::vector<std::pair<int, std::vector<double>>> head_contour_points;
            std::vector<double> thresholds = {0.175, 0.15, 0.125, 0.10, 0.075, 0.05}; // List of thresholds to try
            int segment_count = 0;
            bool success = false;

            for (double threshold_factor : thresholds) {
                head_contour_points.clear();
                sampled_points.clear();
                pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - trying threshold: " << threshold_factor << "\n";

                if (head_at_larger_coord) { // Head on right (x larger)
                    double head_contour_threshold_right = max_x - (max_x - min_x) * threshold_factor;
                    for (const auto& point : fish_points) { if (point.second[0] >= head_contour_threshold_right) head_contour_points.push_back(point); }
                } else { // Head on left (x smaller)
                    double head_contour_threshold_left = min_x + (max_x - min_x) * threshold_factor;
                    for (const auto& point : fish_points) { if (point.second[0] <= head_contour_threshold_left) head_contour_points.push_back(point); }
                }
                
                segment_count = processHeadContour(head_contour_points, sampled_points, struct_id, mpi_rank, fish_points, head_at_larger_coord, false);

                if (segment_count == 5) {
                    success = true;
                    break;
                }
            }

            if (!success) {
                pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - threshold adjustment failed(" << segment_count << " segments), enabling new strategy.\n";
                head_contour_points.clear();
                sampled_points.clear();
                double final_threshold = thresholds.back(); // Use last threshold
                 if (head_at_larger_coord) {
                    double head_contour_threshold_right = max_x - (max_x - min_x) * final_threshold;
                    for (const auto& point : fish_points) { if (point.second[0] >= head_contour_threshold_right) head_contour_points.push_back(point); }
                } else {
                    double head_contour_threshold_left = min_x + (max_x - min_x) * final_threshold;
                    for (const auto& point : fish_points) { if (point.second[0] <= head_contour_threshold_left) head_contour_points.push_back(point); }
                }
                processHeadContour(head_contour_points, sampled_points, struct_id, mpi_rank, fish_points, head_at_larger_coord, true);
            }

        } else { // Vertical swimming
            std::vector<std::pair<int, std::vector<double>>> head_contour_points;
            std::vector<double> thresholds =  {0.175, 0.15, 0.125, 0.10, 0.075, 0.05}; // According to requested threshold order
            int segment_count = 0;
            bool success = false;

            for (double threshold_factor : thresholds) {
                head_contour_points.clear();
                sampled_points.clear();
                pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - trying threshold: " << threshold_factor << "\n";
                
                if (head_at_larger_coord) { // Head on top (y larger)
                    double head_contour_threshold_top = max_y - (max_y - min_y) * threshold_factor;
                    for (const auto& point : fish_points) { if (point.second[1] >= head_contour_threshold_top) head_contour_points.push_back(point); }
                } else { // Head on bottom (y smaller)
                    double head_contour_threshold_bottom = min_y + (max_y - min_y) * threshold_factor;
                    for (const auto& point : fish_points) { if (point.second[1] <= head_contour_threshold_bottom) head_contour_points.push_back(point); }
                }
                
                segment_count = processHeadContourVertical(head_contour_points, sampled_points, struct_id, mpi_rank, fish_points, head_at_larger_coord, false);

                if (segment_count == 5) {
                    success = true;
                    break;
                }
            }

            if (!success) {
                pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - threshold adjustment failed(" << segment_count << " segments), enabling new strategy.\n";
                head_contour_points.clear();
                sampled_points.clear();
                double final_threshold = thresholds.back(); // Use last threshold
                if (head_at_larger_coord) { // Head on top (y larger)
                    double head_contour_threshold_top = max_y - (max_y - min_y) * final_threshold;
                    for (const auto& point : fish_points) { if (point.second[1] >= head_contour_threshold_top) head_contour_points.push_back(point); }
                } else { // Head on bottom (y smaller)
                    double head_contour_threshold_bottom = min_y + (max_y - min_y) * final_threshold;
                    for (const auto& point : fish_points) { if (point.second[1] <= head_contour_threshold_bottom) head_contour_points.push_back(point); }
                }
                processHeadContourVertical(head_contour_points, sampled_points, struct_id, mpi_rank, fish_points, head_at_larger_coord, true);
            }
        }
        
        // 4. Add sampled points to target list
        for (const auto& point : sampled_points) {
            all_target_coords.push_back(point.second);
            target_fish_ids.push_back(struct_id);
            target_point_ids.push_back(point.first);
        }
    }
}

    // Broadcast total number of points to find
    int num_target_points = 0;
    if (mpi_rank == 0) {
        num_target_points = all_target_coords.size();
    }
    MPI_Bcast(&num_target_points, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    
    // If no points to find, return directly
    if (num_target_points == 0) {
        for (size_t i = 0; i < fish_data.size(); ++i) {
            fish_data[i].pressures.clear();
            fish_data[i].coordinates.clear();
        }
        return fish_data;
    }
    
    // Broadcast all point coordinates to find
    std::vector<double> flat_target_coords;
    if (mpi_rank == 0) {
        flat_target_coords.reserve(num_target_points * NDIM);
        for (const auto& coord : all_target_coords) {
            flat_target_coords.insert(flat_target_coords.end(), coord.begin(), coord.end());
        }
    } else {
        flat_target_coords.resize(num_target_points * NDIM);
    }
  

    MPI_Bcast(flat_target_coords.data(), num_target_points * NDIM, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

    
    // Broadcast fish ID and point ID corresponding to each point
    if (mpi_rank != 0) {
        target_fish_ids.resize(num_target_points);
        target_point_ids.resize(num_target_points);
    }
    

    MPI_Bcast(target_fish_ids.data(), num_target_points, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(target_point_ids.data(), num_target_points, MPI_INT, 0, PETSC_COMM_WORLD);
    
    // Each process searches for points on Eulerian grid it manages
    std::vector<bool> local_found(num_target_points, false);
    std::vector<double> local_pressures(num_target_points, 0.0);
    
// Find pressure values corresponding to each point
for (int ln = finest_hier_level; ln >= 0; --ln) {
    Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
    
    // Iterate through all patches at current level
    for (PatchLevel<NDIM>::Iterator p(level); p; p++) {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
        const double* const dx = patch_geom->getDx();
        const double* const patch_x_lower = patch_geom->getXLower();
        const double* const patch_x_upper = patch_geom->getXUpper();
        
        Pointer<CellData<NDIM, double>> p_data = patch->getPatchData(p_idx);
        const Box<NDIM>& box = patch->getBox();
        const hier::Index<NDIM>& box_lower = box.lower();
        
        // Check each target point
        for (int i = 0; i < num_target_points; ++i) {
            if (local_found[i]) continue;
            
            double X[NDIM];
            for (int d = 0; d < NDIM; ++d) {
                X[d] = flat_target_coords[i * NDIM + d];
            }
            
            // Check if point is within current patch
            bool in_patch = true;
            for (int d = 0; d < NDIM; ++d) {
                if (X[d] < patch_x_lower[d] || X[d] >= patch_x_upper[d]) {
                    in_patch = false;
                    break;
                }
            }
            
            if (in_patch) {
                // Calculate cell index
                CellIndex<NDIM> idx;
                for (int d = 0; d < NDIM; ++d) {
                    idx(d) = box_lower(d) + static_cast<int>((X[d] - patch_x_lower[d]) / dx[d]);
                }
                
                // Get pressure value
                if (box.contains(idx)) {
                    local_pressures[i] = (*p_data)(idx);
                    local_found[i] = true;
                }
            }
        }
    }
}
    
    // Merge results found by all processes
    std::vector<bool> global_found(num_target_points, false);
    std::vector<double> global_pressures(num_target_points, 0.0);
    
    // Use MPI_Allreduce for boolean OR operation
    std::vector<int> local_found_int(num_target_points, 0);
    std::vector<int> global_found_int(num_target_points, 0);
    for (int i = 0; i < num_target_points; ++i) {
        local_found_int[i] = local_found[i] ? 1 : 0;
    }
    
    MPI_Allreduce(local_found_int.data(), global_found_int.data(), num_target_points, MPI_INT, MPI_LOR, PETSC_COMM_WORLD);
    
    for (int i = 0; i < num_target_points; ++i) {
        global_found[i] = (global_found_int[i] != 0);
    }
    
    // Create temporary array, multiply found pressures by found flag
    std::vector<double> local_pressures_weighted(num_target_points, 0.0);
    for (int i = 0; i < num_target_points; ++i) {
        if (local_found[i]) {
            local_pressures_weighted[i] = local_pressures[i];
        }
    }
    
    MPI_Allreduce(local_pressures_weighted.data(), global_pressures.data(), num_target_points, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    
    // Initialize fish_data structure
    for (size_t i = 0; i < fish_data.size(); ++i) {
        fish_data[i].pressures.clear();
        fish_data[i].coordinates.clear();
    }
    
    // Organize results by fish
    for (int i = 0; i < num_target_points; ++i) {
        int fish_id = target_fish_ids[i];
        
        std::vector<double> coord(NDIM);
        for (int d = 0; d < NDIM; ++d) {
            coord[d] = flat_target_coords[i * NDIM + d];
        }
        
        double pressure = global_found[i] ? global_pressures[i] : 0.0;
        
        fish_data[fish_id].pressures.push_back(pressure);
        fish_data[fish_id].coordinates.push_back(coord);
    }
    

    return fish_data;
}

// Modified Point structure name to FishPoint
struct FishPoint {
    double x, y;  // 2D coordinates
    int original_idx;  // Original index
    int cluster_id;    // Cluster ID

    FishPoint(double x, double y, int idx) : x(x), y(y), original_idx(idx), cluster_id(-1) {}
};

// Modified distance function
double distance(const FishPoint& p1, const FishPoint& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// Modified clustering function, cluster after gathering all process data
std::vector<std::vector<int>> clusterFishByCoordinates(
    Pointer<PatchHierarchy<NDIM>> patch_hierarchy,
    LDataManager* l_data_manager) {

    // Get MPI communicator and process information
    const int mpi_rank = IBTK_MPI::getRank();
    const int mpi_size = IBTK_MPI::getNodes();
    
    // Get finest hierarchy level
    const int finest_hier_level = patch_hierarchy->getFinestLevelNumber();
    
    // Get Lagrangian data
    Pointer<LData> X_data = l_data_manager->getLData("X", finest_hier_level);
    
    // Get local Lagrangian point count
    const int local_node_count = X_data->getLocalNodeCount();
    
    // Get Lagrangian point coordinate data
    double* X_data_array = X_data->getLocalFormVecArray()->data();
    
    // Create local point set
    std::vector<double> local_coords;
    std::vector<int> local_indices;
    
    // Fill local coordinates and indices
    for (int i = 0; i < local_node_count; ++i) {
        double x = X_data_array[i * NDIM];
        double y = X_data_array[i * NDIM + 1];
        local_coords.push_back(x);
        local_coords.push_back(y);
        local_indices.push_back(i);
    }
    
    // Release Lagrangian data array
    X_data->restoreArrays();
    
    // First collect how many points each process has
    std::vector<int> all_counts(mpi_size, 0);
    int local_count = local_node_count;
    MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);
    
    // Calculate offsets
    std::vector<int> displacements(mpi_size, 0);
    int total_count = 0;
    for (int i = 0; i < mpi_size; ++i) {
        displacements[i] = total_count;
        total_count += all_counts[i];
    }
    
    // Create arrays for saving all point coordinates and indices
    std::vector<double> all_coords;
    std::vector<int> all_indices;
    std::vector<int> all_global_indices;
    
        if (mpi_rank == 0) {
        all_coords.resize(total_count * NDIM);
        all_indices.resize(total_count);
        all_global_indices.resize(total_count);
        }
        
    // Collect all coordinates
        std::vector<int> coord_counts(mpi_size);
        std::vector<int> coord_displacements(mpi_size);
        for (int i = 0; i < mpi_size; ++i) {
        coord_counts[i] = all_counts[i] * NDIM;
            coord_displacements[i] = displacements[i] * NDIM;
        }
        
    MPI_Gatherv(local_coords.data(), local_count * NDIM, MPI_DOUBLE,
               all_coords.data(), coord_counts.data(), coord_displacements.data(),
                   MPI_DOUBLE, 0, PETSC_COMM_WORLD);
        
    // Calculate global indices
    std::vector<int> global_indices(local_count);
    for (int i = 0; i < local_count; ++i) {
        // Calculate this point's index in global array
        global_indices[i] = displacements[mpi_rank] + i;
    }
    
    // Collect all global indices
    MPI_Gatherv(global_indices.data(), local_count, MPI_INT,
               all_global_indices.data(), all_counts.data(), displacements.data(),
                   MPI_INT, 0, PETSC_COMM_WORLD);
        
    // Collect local indices (for later recovery)
    MPI_Gatherv(local_indices.data(), local_count, MPI_INT,
               all_indices.data(), all_counts.data(), displacements.data(),
               MPI_INT, 0, PETSC_COMM_WORLD);
    
    // Only execute clustering on root process
    std::vector<std::vector<int>> all_clusters;
    
        if (mpi_rank == 0) {
        // Create point set
        std::vector<FishPoint> points;
        points.reserve(total_count);
        
        // Fill point set
        for (int i = 0; i < total_count; ++i) {
            double x = all_coords[i * NDIM];
            double y = all_coords[i * NDIM + 1];
            points.emplace_back(x, y, all_global_indices[i]);
        }
        
        // DBSCAN parameter settings
        const double eps = 0.2;     // Neighborhood radius
        const int min_pts = 5;      // Minimum neighbor count
        
        // Execute DBSCAN clustering... (this part is long, complete DBSCAN implementation)
        int current_cluster = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            if (points[i].cluster_id != -1) continue;
            std::vector<size_t> neighbors;
            for (size_t j = 0; j < points.size(); ++j) {
                if (i != j && distance(points[i], points[j]) < eps) {
                    neighbors.push_back(j);
                }
            }
            if (neighbors.size() < min_pts) {
                points[i].cluster_id = -1;
                continue;
            }
            current_cluster++;
            points[i].cluster_id = current_cluster;
            std::queue<size_t> queue;
            for (const auto& neighbor : neighbors) queue.push(neighbor);
            while (!queue.empty()) {
                size_t current = queue.front();
                queue.pop();
                if (points[current].cluster_id == -1) {
                    points[current].cluster_id = current_cluster;
                    std::vector<size_t> sub_neighbors;
                    for (size_t j = 0; j < points.size(); ++j) {
                        if (current != j && distance(points[current], points[j]) < eps) {
                            sub_neighbors.push_back(j);
                        }
                    }
                    if (sub_neighbors.size() >= min_pts) {
                        for (const auto& sub_neighbor : sub_neighbors) {
                            if (points[sub_neighbor].cluster_id == -1) {
                                queue.push(sub_neighbor);
                            }
                        }
                    }
                }
                if (points[current].cluster_id > 0 && points[current].cluster_id != current_cluster) {
                    continue;
                }
            }
        }
        
        // Generate point index sets for each cluster
        std::unordered_map<int, std::vector<int>> clusters;
        for (const auto& point : points) {
            if (point.cluster_id > 0) {
                clusters[point.cluster_id].push_back(point.original_idx);
            }
        }
        
        // Edge detection: only keep edge points for each cluster
        std::unordered_map<int, std::vector<int>> edge_clusters;
        
        for (const auto& cluster_pair : clusters) {
            int cluster_id = cluster_pair.first;
            const auto& cluster_points_idx = cluster_pair.second;
            std::unordered_map<int, std::pair<double, double>> point_coords;
            for (const int& idx : cluster_points_idx) {
                for (int j = 0; j < total_count; ++j) {
                    if (all_global_indices[j] == idx) {
                        point_coords[idx] = std::make_pair(all_coords[j * NDIM], all_coords[j * NDIM + 1]);
                        break;
                    }
                }
            }
            
            // Edge detection parameters
            const double search_radius = 0.3;
            const int angle_divisions = 8;
            const int min_empty_sectors = 2;
            std::vector<int> edge_points;
            
            for (const int& idx : cluster_points_idx) {
                auto& center = point_coords[idx];
                std::vector<int> sector_counts(angle_divisions, 0);
                for (const int& other_idx : cluster_points_idx) {
                    if (other_idx == idx) continue;
                    auto& other = point_coords[other_idx];
                    double dx = other.first - center.first;
                    double dy = other.second - center.second;
                    double dist = std::sqrt(dx * dx + dy * dy);
                    if (dist <= search_radius) {
                        double angle = std::atan2(dy, dx);
                        if (angle < 0) angle += 2 * M_PI;
                        int sector = static_cast<int>(angle / (2 * M_PI / angle_divisions));
                        sector_counts[sector]++;
                    }
                }
                int empty_sectors = 0;
                for (int count : sector_counts) if (count == 0) empty_sectors++;
                if (empty_sectors >= min_empty_sectors) edge_points.push_back(idx);
            }
            edge_clusters[cluster_id] = edge_points;
        }

        // New filtering logic starts here
        
        std::vector<std::vector<int>> fish_only_clusters;
        
        for (const auto& cluster_pair : edge_clusters) {
            const std::vector<int>& cluster_points = cluster_pair.second;
            
            if (cluster_points.empty()) continue; // Skip empty clusters
            
            // Calculate cluster center for debugging
            double center_x = 0, center_y = 0;
            for (int global_idx : cluster_points) {
                for (int j = 0; j < total_count; ++j) {
                    if (all_global_indices[j] == global_idx) {
                        center_x += all_coords[j * NDIM];
                        center_y += all_coords[j * NDIM + 1];
                        break;
                    }
                }
            }
            center_x /= cluster_points.size();
            center_y /= cluster_points.size();
            
            pout << "Cluster " << cluster_pair.first << " point count: " << cluster_points.size() 
                 << ", center: (" << center_x << ", " << center_y << ")";
             
            // Determine if it's a fish body: point count > 250
            if (cluster_points.size() > 250) {
                fish_only_clusters.push_back(cluster_points);
                pout << " -> Identified as fish body" << std::endl;
            } else {
                pout << " -> Identified as cylinder or other structure" << std::endl;
            }
        }
        
        // Update all_clusters to only contain fish clusters
        all_clusters = fish_only_clusters;
        
        pout << "Clustering completed, found " << all_clusters.size() << " fish" << std::endl;
        for (size_t i = 0; i < all_clusters.size(); ++i) {
            pout << "  Fish " << i << " contains " << all_clusters[i].size() << " edge points" << std::endl;
        }
    }

    // Broadcast cluster count to all processes
    int num_clusters = 0;
    if (mpi_rank == 0) {
        num_clusters = all_clusters.size();
    }
    MPI_Bcast(&num_clusters, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    
    // Broadcast cluster data to all processes
    if (mpi_rank != 0) {
        all_clusters.resize(num_clusters);
    }
    
    for (int i = 0; i < num_clusters; ++i) {
        int cluster_size = 0;
        if (mpi_rank == 0) {
            cluster_size = all_clusters[i].size();
        }
        MPI_Bcast(&cluster_size, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        
        if (mpi_rank != 0) {
            all_clusters[i].resize(cluster_size);
        }
        
        MPI_Bcast(all_clusters[i].data(), cluster_size, MPI_INT, 0, PETSC_COMM_WORLD);
    }

    return all_clusters;
}

/*******************************************************************************
 * For each run, the input filename and restart information (if needed) must   *
 * be given on the command line.  For non-restarted case, command line is:     *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 * For restarted run, command line is:                                         *
 *                                                                             *
 *    executable <input file name> <restart directory> <restart number>        *
 *                                                                             *
 *******************************************************************************/
int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object as well.
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Read fish activation status from input file
        // Set ACTIVATE_FISH_IN_THIS_RUN in input2d to manually control fish activation
        // - false: fish deactivated for pure cylinder flow calculation
        // - true:  fish activated for coupled calculation (usually from restart file)
        bool ACTIVATE_FISH_IN_THIS_RUN = false;
        if (input_db->keyExists("Main"))
        {
            ACTIVATE_FISH_IN_THIS_RUN = input_db->getDatabase("Main")->getBoolWithDefault("ACTIVATE_FISH_IN_THIS_RUN", false);
        }

        // Read Python server port from input file
        int py_server_port = 9999;
        if (input_db->keyExists("Main"))
        {
            py_server_port = input_db->getDatabase("Main")->getIntegerWithDefault("py_server_port", 9999);
        }
        if (IBTK_MPI::getRank() == 0) {
            pout << "Will connect to Python server port: " << py_server_port << "\n";
        }

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && !app_initializer->getVisItDataWriter().isNull();

        const bool dump_restart_data = app_initializer->dumpRestartData();
        const int restart_dump_interval = app_initializer->getRestartDumpInterval();
        const string restart_dump_dirname = app_initializer->getRestartDumpDirectory();

        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const int postproc_data_dump_interval = app_initializer->getPostProcessingDataDumpInterval();
        const string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && (postproc_data_dump_interval > 0) && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<INSHierarchyIntegrator> navier_stokes_integrator = new INSStaggeredHierarchyIntegrator(
            "INSStaggeredHierarchyIntegrator",
            app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));

        const int num_structures = input_db->getIntegerWithDefault("num_structures", 1);
        Pointer<ConstraintIBMethod> ib_method_ops = new ConstraintIBMethod(
            "ConstraintIBMethod", app_initializer->getComponentDatabase("ConstraintIBMethod"), num_structures);
        Pointer<IBHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                              app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                                              ib_method_ops,
                                              navier_stokes_integrator);

        Pointer<CartesianGridGeometry<NDIM> > grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM> > patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);

        Pointer<StandardTagAndInitialize<NDIM> > error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
                                               app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM> > box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM> > load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM> > gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Configure the IB solver.
        Pointer<IBStandardInitializer> ib_initializer = new IBStandardInitializer(
            "IBStandardInitializer", app_initializer->getComponentDatabase("IBStandardInitializer"));
        ib_method_ops->registerLInitStrategy(ib_initializer);
        Pointer<IBStandardForceGen> ib_force_fcn = new IBStandardForceGen();
        ib_method_ops->registerIBLagrangianForceFunction(ib_force_fcn);

        // Create Eulerian initial condition specification objects.
        if (input_db->keyExists("VelocityInitialConditions"))
        {
            Pointer<CartGridFunction> u_init = new muParserCartGridFunction(
                "u_init", app_initializer->getComponentDatabase("VelocityInitialConditions"), grid_geometry);
            navier_stokes_integrator->registerVelocityInitialConditions(u_init);
        }

        if (input_db->keyExists("PressureInitialConditions"))
        {
            Pointer<CartGridFunction> p_init = new muParserCartGridFunction(
                "p_init", app_initializer->getComponentDatabase("PressureInitialConditions"), grid_geometry);
            navier_stokes_integrator->registerPressureInitialConditions(p_init);
        }

        // Create Eulerian boundary condition specification objects (when necessary).
        const IntVector<NDIM>& periodic_shift = grid_geometry->getPeriodicShift();
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        if (periodic_shift.min() > 0)
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                u_bc_coefs[d] = nullptr;
            }
        }
        else
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const std::string bc_coefs_name = "u_bc_coefs_" + std::to_string(d);

                const std::string bc_coefs_db_name = "VelocityBcCoefs_" + std::to_string(d);

                u_bc_coefs[d] = new muParserRobinBcCoefs(
                    bc_coefs_name, app_initializer->getComponentDatabase(bc_coefs_db_name), grid_geometry);
            }
            navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);
        }

        // Create Eulerian body force function specification objects.
        if (input_db->keyExists("ForcingFunction"))
        {
            Pointer<CartGridFunction> f_fcn = new muParserCartGridFunction(
                "f_fcn", app_initializer->getComponentDatabase("ForcingFunction"), grid_geometry);
            time_integrator->registerBodyForceFunction(f_fcn);
        }

        // Set up visualization plot file writers.
        Pointer<VisItDataWriter<NDIM> > visit_data_writer = app_initializer->getVisItDataWriter();
        Pointer<LSiloDataWriter> silo_data_writer = app_initializer->getLSiloDataWriter();
        if (uses_visit)
        {
            ib_initializer->registerLSiloDataWriter(silo_data_writer);
            ib_method_ops->registerLSiloDataWriter(silo_data_writer);
            time_integrator->registerVisItDataWriter(visit_data_writer);
        }

        // Initialize hierarchy configuration and data on all patches.
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);


        // Create ConstraintIBKinematics objects
        vector<Pointer<ConstraintIBKinematics> > ibkinematics_ops_vec;
        Pointer<ConstraintIBKinematics> ib_kinematics_op;
        // struct_0: fish
        pout << "Preparing to create IBEELKinematics object (fish)...\n";
        ib_kinematics_op =
            new IBEELKinematics("eel2d",
                                app_initializer->getComponentDatabase("ConstraintIBKinematics")->getDatabase("eel2d"),
                                ib_method_ops->getLDataManager(),
                                patch_hierarchy);
        ibkinematics_ops_vec.push_back(ib_kinematics_op);
        pout << "IBEELKinematics object creation completed\n";
        
        // struct_1: cylinder
        pout << "Preparing to create IBCylinderKinematics object (cylinder)...\n";
        ib_kinematics_op =
            new IBCylinderKinematics("cylinder2d",
                                    app_initializer->getComponentDatabase("ConstraintIBKinematics")->getDatabase("cylinder2d"),
                                    ib_method_ops->getLDataManager(),
                                    patch_hierarchy);
        ibkinematics_ops_vec.push_back(ib_kinematics_op);
        pout << "IBCylinderKinematics object creation completed\n";

        // register ConstraintIBKinematics objects with ConstraintIBMethod.
        ib_method_ops->registerConstraintIBKinematics(ibkinematics_ops_vec);
        ib_method_ops->initializeHierarchyOperatorsandData();

        // Create hydrodynamic force evaluator object.
        double rho_fluid = input_db->getDouble("RHO");
        double mu_fluid = input_db->getDouble("MU");
        double start_time = time_integrator->getIntegratorTime();
        Pointer<IBHydrodynamicForceEvaluator> hydro_force =
            new IBHydrodynamicForceEvaluator("IBHydrodynamicForce", rho_fluid, mu_fluid, start_time, true);

        // Get the initial box position and velocity from input for first fish
        const string init_hydro_force_box_db_name = "InitHydroForceBox_0";
        IBTK::Vector3d box_X_lower, box_X_upper, box_init_vel;

        input_db->getDatabase(init_hydro_force_box_db_name)->getDoubleArray("lower_left_corner", &box_X_lower[0], 3);
        input_db->getDatabase(init_hydro_force_box_db_name)->getDoubleArray("upper_right_corner", &box_X_upper[0], 3);
        input_db->getDatabase(init_hydro_force_box_db_name)->getDoubleArray("init_velocity", &box_init_vel[0], 3);

        // Register control volume for first fish
        hydro_force->registerStructure(box_X_lower, box_X_upper, patch_hierarchy, box_init_vel, 0);

        // Set torque origin and register plot data for fish (structure 0)
        std::vector<std::vector<double>> structure_COM = ib_method_ops->getCurrentStructureCOM();
        IBTK::Vector3d eel_COM_1;
        for (int d = 0; d < 3; ++d) {
            eel_COM_1[d] = structure_COM[0][d];
        }
        hydro_force->setTorqueOrigin(eel_COM_1, 0);
        hydro_force->registerStructurePlotData(visit_data_writer, patch_hierarchy, 0);

        // Deallocate initialization objects.
        ib_method_ops->freeLInitStrategy();
        ib_initializer.setNull();
        app_initializer.setNull();

        // ---------------------------------------------------------------------
        // Manual control: activate/deactivate fish based on input parameter
        //
        if (!ACTIVATE_FISH_IN_THIS_RUN)
        {
            // Phase 1: generate pure cylinder flow field
            pout << "\n#####################################################\n";
            pout << "MANUAL CONTROL: Deactivating structure 0 (fish) for this run.\n";
            pout << "#####################################################\n\n";
            const std::vector<int> fish_struct_id = {0};
            for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
            {
                ib_method_ops->getLDataManager()->inactivateLagrangianStructures(fish_struct_id, ln);
            }
        }
        else
        {
            // Phase 2: perform coupled calculation (usually from restart file)
            // When starting from a restart file with deactivated fish, explicitly activate it
            pout << "\n#####################################################\n";
            pout << "MANUAL CONTROL: Activating structure 0 (fish) for this run.\n";
            pout << "#####################################################\n\n";
            const std::vector<int> fish_struct_id = {0};
            for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
            {
                // Safe to call even if structure is already active
                ib_method_ops->getLDataManager()->activateLagrangianStructures(fish_struct_id, ln);
            }
        }
        // ---------------------------------------------------------------------

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Get velocity and pressure variables from integrator
        VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();

        const Pointer<Variable<NDIM> > u_var = navier_stokes_integrator->getVelocityVariable();
        const Pointer<VariableContext> u_ctx = navier_stokes_integrator->getCurrentContext();
        const int u_idx = var_db->mapVariableAndContextToIndex(u_var, u_ctx);

        const Pointer<Variable<NDIM> > p_var = navier_stokes_integrator->getPressureVariable();
        const Pointer<VariableContext> p_ctx = navier_stokes_integrator->getCurrentContext();
        const int p_idx = var_db->mapVariableAndContextToIndex(p_var, p_ctx);

        // Write out initial visualization data.
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        if (dump_viz_data && uses_visit)
        {
            pout << "\n\nWriting visualization files...\n\n";
            time_integrator->setupPlotData();
            visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            silo_data_writer->writePlotData(iteration_num, loop_time);
        }

        // Main time step loop.
        double loop_time_end = time_integrator->getEndTime();
        double dt = 0.0;
        double current_time, new_time;
        double box_disp = 0.0;

        // Initialize motion control parameters
        double theta_lmax_prev = 1.0;  // Previous theta_lmax value
        double theta_lmax_next = -1.0; // Next theta_lmax value  
        double lambda_prev = 1.0;      // Previous wavelength
        double lambda_next = 1.0;      // Next wavelength
        double last_coeff_update_time = input_db->getDoubleWithDefault("LAST_COEFF_UPDATE_TIME", 0.0);
        
        if (IBTK_MPI::getRank() == 0) {
            pout << "=== Parameter Reading Verification ===\n";
            pout << "LAST_COEFF_UPDATE_TIME from input file = " << last_coeff_update_time << "\n";
            pout << "=====================================\n";
        }
        double dynamic_coeff_update_interval = 0.5;   // Coefficient update interval

        // Container for caching state records
        std::vector<StateRecord> state_record_cache;
        static double last_state_record_time = 0.0;
        static const double state_record_interval = 0.05;  // Record state every 0.05s

        while (!IBTK::rel_equal_eps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time = time_integrator->getIntegratorTime();
            current_time = loop_time;

            // Output current timestep info (every 0.05s)
            static double last_output_time = -0.01;
            if (floor(loop_time/0.05) > floor(last_output_time/0.05)) {
                pout << "\n";
                pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
                pout << "Current timestep # " << iteration_num << "\n";
                pout << "Simulation time is " << loop_time << "\n";
          
                last_output_time = loop_time;
            }

            dt = time_integrator->getMaximumTimeStepSize();
            
            // Calculate next time
            double next_time = loop_time + dt;
            
            // Regrid the hierarchy if necessary.
            if (time_integrator->atRegridPoint()) time_integrator->regridHierarchy();

            // Set the box velocity to nonzero only if the eel has moved sufficiently far.
            IBTK::Vector3d box_vel;
            box_vel.setZero();

            // Velocity due to free-swimming (for the fish, structure 0)
            std::vector<std::vector<double> > COM_vel = ib_method_ops->getCurrentCOMVelocity();
            for (int d = 0; d < NDIM; ++d) box_vel(d) = COM_vel[0][d];

            int coarsest_ln = 0;
            Pointer<PatchLevel<NDIM> > coarsest_level = patch_hierarchy->getPatchLevel(coarsest_ln);
            const Pointer<CartesianGridGeometry<NDIM> > coarsest_grid_geom = coarsest_level->getGridGeometry();
            const double* const DX = coarsest_grid_geom->getDx();

            // Set the box velocity to ensure that the immersed body remains inside the control volume at all times.
            // If the body's COM has moved 0.9 coarse mesh widths in the x-direction, set the CV velocity such that
            // the CV will translate by 1 coarse mesh width in the direction of swimming (negative x-direction).
            // Otherwise, keep the CV in place by setting its velocity to zero.

            box_disp += box_vel[0] * dt;
            if (abs(box_disp) >= abs(0.9 * DX[0]))
            {
                box_vel.setZero();
                box_vel[0] = -DX[0] / dt;

                box_disp = 0.0;
            }
            else
            {
                box_vel.setZero();
            }

            // Update the location of the box for time n + 1 (for the fish)
            hydro_force->updateStructureDomain(box_vel, dt, patch_hierarchy, 0);

            // Compute the momentum of u^n in box n+1 on the newest hierarchy
            hydro_force->computeLaggedMomentumIntegral(
                u_idx, patch_hierarchy, navier_stokes_integrator->getVelocityBoundaryConditions());

            // Advance the hierarchy
            time_integrator->advanceHierarchy(dt);
            
            // Update loop_time to next time value
            loop_time = next_time;
            new_time = loop_time;

            if (current_time < new_time && 
                (new_time - last_coeff_update_time) >= dynamic_coeff_update_interval) {
                
                // Send cached state data before requesting control
                if (IBTK_MPI::getRank() == 0 && !state_record_cache.empty()) {
                    send_state_record_batch(state_record_cache, py_server_port);
                    state_record_cache.clear();
                    pout << "Cached state records sent and cache cleared.\n";
                }
                
                // Collect fish pressure data (clustering algorithm automatically identifies fish)
                std::vector<FishPressureData> fish_data = collect_fish_pressure_data(
                    patch_hierarchy,
                    navier_stokes_integrator,
                    ib_method_ops->getLDataManager(),
                    p_idx,
                    1,  // Only focus on fish
                    ib_method_ops
                );
                
                // Call batch processing function (only main process executes)
                ControllerResult controller_result = call_python_controllers_batch(fish_data, new_time, ib_method_ops, py_server_port);
                
                // Broadcast results to all processes
                if (!fish_data.empty()) {
                    MPI_Bcast(controller_result.amplitudes.data(), 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
                    MPI_Bcast(controller_result.T_n_values.data(), 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
                }
                
                // Record current state before update and print details
                if (IBTK_MPI::getRank() == 0) {
                    pout << "\n########## New parameters from Python server (time=" << new_time 
                         << ", interval=" << dynamic_coeff_update_interval << ") ##########\n";
                    pout << "Last update time: " << last_coeff_update_time 
                         << ", current time: " << new_time 
                         << ", time difference: " << (new_time - last_coeff_update_time) << "\n";
                    pout << "Current coefficients: theta_lmax_prev=" << theta_lmax_prev 
                         << ", theta_lmax_next=" << theta_lmax_next
                         << ", lambda_prev=" << lambda_prev
                         << ", lambda_next=" << lambda_next << "\n";
                }
                
                // Update coefficients: assign previous theta_lmax_next to theta_lmax_prev
                theta_lmax_prev = theta_lmax_next;
                
                // Get amplitude value from Python (always positive)
                double amplitude_magnitude = fish_data.empty() ? 1.0 : controller_result.amplitudes[0];

                // Ensure new theta_lmax_next has opposite sign to theta_lmax_prev
                if (theta_lmax_prev > 0) {
                    theta_lmax_next = -amplitude_magnitude;  // Set to negative
                } else {
                    theta_lmax_next = amplitude_magnitude;   // Set to positive
                }
                
                // Update T_n
                double new_T_n = fish_data.empty() ? 1.0 : controller_result.T_n_values[0];
                
                // Apply calculation results - only process fish (structure 0)
                IBEELKinematics* eel_kinematics = 
                    dynamic_cast<IBEELKinematics*>(ibkinematics_ops_vec[0].getPointer());
                if (eel_kinematics) {
                    // Call updateCoefficients method to update motion parameters
                    eel_kinematics->updateCoefficients(theta_lmax_prev, theta_lmax_next, 
                                 lambda_prev, lambda_next, 
                                 current_time, new_T_n);
                        
                    // Update dynamic update interval to half of new T_n
                    dynamic_coeff_update_interval = new_T_n / 2.0; 
                        
                    // Only output info on main process
                    if (IBTK_MPI::getRank() == 0 && !fish_data.empty()) {
                        pout << "Time " << new_time 
                             << " Fish " << 0 
                             << " pressure points " << fish_data[0].pressures.size()
                             << " RL returned amplitude=" << amplitude_magnitude
                             << " theta_lmax_prev=" << theta_lmax_prev
                             << " new theta_lmax_next=" << theta_lmax_next
                             << " new T_n=" << new_T_n
                             << " new update interval=" << dynamic_coeff_update_interval
                             << " coefficients updated\n";
                    }
                }
                
                // Update last update time
                last_coeff_update_time = new_time;
                
                if (IBTK_MPI::getRank() == 0) {
                    pout << "Updated coefficients: theta_lmax_prev=" << theta_lmax_prev 
                         << ", theta_lmax_next=" << theta_lmax_next
                         << ", lambda_prev=" << lambda_prev
                         << ", lambda_next=" << lambda_next << "\n";
                    pout << "Next update time: " << (last_coeff_update_time + dynamic_coeff_update_interval) << "\n";
                    pout << "########## Python server parameter update completed ##########\n\n";
                }
            }

            // Get the momentum of the eel (structure 0)
            IBTK::Vector3d eel_mom, eel_rot_mom;
            eel_mom.setZero();
            eel_rot_mom.setZero();
            std::vector<std::vector<double> > structure_linear_momentum = ib_method_ops->getStructureMomentum();
            for (int d = 0; d < NDIM; ++d) eel_mom[d] = structure_linear_momentum[0][d];
            std::vector<std::vector<double> > structure_rotational_momentum =
                ib_method_ops->getStructureRotationalMomentum();
            for (int d = 0; d < 3; ++d) eel_rot_mom[d] = structure_rotational_momentum[0][d];

            // Store the new momenta of the eel
            hydro_force->updateStructureMomentum(eel_mom, eel_rot_mom, 0);

            // Evaluate hydrodynamic force on the structures.
            hydro_force->computeHydrodynamicForce(u_idx,
                                                  p_idx,
                                                  /*f_idx*/ -1,
                                                  patch_hierarchy,
                                                  dt,
                                                  navier_stokes_integrator->getVelocityBoundaryConditions(),
                                                  navier_stokes_integrator->getPressureBoundaryConditions());

            // Print the drag and torque
            hydro_force->postprocessIntegrateData(current_time, new_time);

            // Update CV plot data for the fish
            hydro_force->updateStructurePlotData(patch_hierarchy, 0);

            // Set the torque origin for the next time step (for the fish)
            structure_COM = ib_method_ops->getCurrentStructureCOM();
            for (int d = 0; d < 3; ++d) {
                eel_COM_1[d] = structure_COM[0][d];
            }

            // Only update torque calculation axis for first fish
            hydro_force->setTorqueOrigin(eel_COM_1, 0);

            // At specified intervals, write visualization and restart files,
            // print out timer data, and store hierarchy data for post
            // processing.
            iteration_num += 1;
            const bool last_step = !time_integrator->stepsRemaining();
            if (dump_viz_data && uses_visit && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                pout << "\nWriting visualization files...\n\n";
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                silo_data_writer->writePlotData(iteration_num, loop_time);
            }
            if (dump_restart_data && (iteration_num % restart_dump_interval == 0 || last_step))
            {
                pout << "\nWriting restart files...\n\n";
                RestartManager::getManager()->writeRestartFile(restart_dump_dirname, iteration_num);
            }
            if (dump_timer_data && (iteration_num % timer_dump_interval == 0 || last_step))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
            }
            if (dump_postproc_data && (iteration_num % postproc_data_dump_interval == 0 || last_step))
            {
                output_data(patch_hierarchy,
                           navier_stokes_integrator,
                           ib_method_ops->getLDataManager(),
                           iteration_num,
                           loop_time,
                           postproc_data_dump_dirname);
            }

            // Cache state info every 0.05s instead of sending directly
            if ((new_time - last_state_record_time) >= state_record_interval) {
                // All processes must call this function due to MPI collective operations
                std::vector<FishPressureData> state_fish_data = collect_fish_pressure_data(
                    patch_hierarchy,
                    navier_stokes_integrator,
                    ib_method_ops->getLDataManager(),
                    p_idx,
                    1,  // Only focus on fish
                    ib_method_ops
                );

                if (IBTK_MPI::getRank() == 0) { // Only cache data on main process
                    if (!state_fish_data.empty()) {
                        state_record_cache.push_back({
                            new_time,
                            state_fish_data,
                            ib_method_ops->getInertiaPower(),
                            ib_method_ops->getConstraintPower(),
                            ib_method_ops->getRigidTranslationalVelocity(),
                            ib_method_ops->getStructureRotationalMomentum()
                        });
                        pout << "State cached: time=" << new_time << ", current cache size: " << state_record_cache.size() << "\n";
                    }
                }
                last_state_record_time = new_time;
            }
        }

        // Cleanup Eulerian boundary condition specification objects (when
        // necessary).
        for (unsigned int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];

        // Clean up Python interpreter
        finalize_python();

    } // cleanup dynamically allocated objects prior to shutdown

    return 0;
} // main

void
output_data(Pointer<PatchHierarchy<NDIM> > patch_hierarchy,
            Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
            LDataManager* l_data_manager,
            const int iteration_num,
            const double loop_time,
            const string& data_dump_dirname)
{
    plog << "writing hierarchy data at iteration " << iteration_num << " to disk" << endl;
    plog << "simulation time is " << loop_time << endl;

    // Write Cartesian data.
    string file_name = data_dump_dirname + "/" + "hier_data.";
    char temp_buf[128];
    sprintf(temp_buf, "%05d.samrai.%05d", iteration_num, IBTK_MPI::getRank());
    file_name += temp_buf;
    Pointer<HDFDatabase> hier_db = new HDFDatabase("hier_db");
    hier_db->create(file_name);
    VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
    ComponentSelector hier_data;
    hier_data.setFlag(var_db->mapVariableAndContextToIndex(navier_stokes_integrator->getVelocityVariable(),
                                                           navier_stokes_integrator->getCurrentContext()));
    hier_data.setFlag(var_db->mapVariableAndContextToIndex(navier_stokes_integrator->getPressureVariable(),
                                                           navier_stokes_integrator->getCurrentContext()));
    patch_hierarchy->putToDatabase(hier_db->putDatabase("PatchHierarchy"), hier_data);
    hier_db->putDouble("loop_time", loop_time);
    hier_db->putInteger("iteration_num", iteration_num);
    hier_db->close();

    const Pointer<Variable<NDIM> > p_var = navier_stokes_integrator->getPressureVariable();
    const Pointer<VariableContext> p_ctx = navier_stokes_integrator->getCurrentContext();
    const int p_idx = var_db->mapVariableAndContextToIndex(p_var, p_ctx);

    // Write Lagrangian data.
    const int finest_hier_level = patch_hierarchy->getFinestLevelNumber();
    Pointer<LData> X_data = l_data_manager->getLData("X", finest_hier_level);
    Vec X_petsc_vec = X_data->getVec();
    Vec X_lag_vec;
    VecDuplicate(X_petsc_vec, &X_lag_vec);
    l_data_manager->scatterPETScToLagrangian(X_petsc_vec, X_lag_vec, finest_hier_level);
    file_name = data_dump_dirname + "/" + "X.";
    sprintf(temp_buf, "%05d", iteration_num);
    file_name += temp_buf;
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, file_name.c_str(), &viewer);
    VecView(X_lag_vec, viewer);
    PetscViewerDestroy(&viewer);
    VecDestroy(&X_lag_vec);
    return;

} // output_data

// Batch send state records function
void send_state_record_batch(
    const std::vector<StateRecord>& record_cache,
    int server_port) {
    
    if (IBTK_MPI::getRank() != 0 || record_cache.empty()) return;

    int sockfd = -1;
    try {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) return;

        struct sockaddr_in serv_addr;
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(server_port);
        const char* head_node_ip = std::getenv("HEAD_NODE_IP");
        if (head_node_ip == nullptr) {
            head_node_ip = "127.0.0.1";
        }
        serv_addr.sin_addr.s_addr = inet_addr(head_node_ip);

        if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            close(sockfd);
            return;
        }

        json request;
        request["request_type"] = "state_record_batch";
        
        char* cwd = getcwd(NULL, 0);
        std::string current_dir(cwd);
        free(cwd);
        size_t pos = current_dir.find_last_of('/');
        if (pos != std::string::npos) {
            request["env_id"] = current_dir.substr(pos + 1);
        } else {
            request["env_id"] = "unknown_env";
        }

        request["records"] = json::array();
        for (const auto& record : record_cache) {
            json record_json;
            record_json["time"] = record.time;

            // Add fish data
            record_json["fish_data"] = json::array();
            for (const auto& fish : record.fish_data) {
                json fish_json;
                fish_json["coordinates"] = fish.coordinates;
                fish_json["pressures"] = fish.pressures;
                record_json["fish_data"].push_back(fish_json);
            }

            // Add power data
            record_json["power_data"] = json::array();
            if (!record.inertia_power.empty() && !record.constraint_power.empty()) {
                json power_json;
                power_json["inertia"] = {{"x", record.inertia_power[0][0]}, {"y", record.inertia_power[0][1]}};
                power_json["constraint"] = {{"x", record.constraint_power[0][0]}, {"y", record.constraint_power[0][1]}};
                record_json["power_data"].push_back(power_json);
            }

            // Add velocity data
            record_json["velocities"] = json::array();
            if (!record.velocities.empty()) {
                json vel_json = {{"x", record.velocities[0][0]}, {"y", record.velocities[0][1]}};
                record_json["velocities"].push_back(vel_json);
            }

            // Add angular velocity data
            record_json["angular_velocities"] = json::array();
            if (!record.angular_velocities.empty()) {
                json rot_vel_json = {{"z", record.angular_velocities[0][2]}};
                record_json["angular_velocities"].push_back(rot_vel_json);
            }
            request["records"].push_back(record_json);
        }

        std::string json_str = request.dump() + "\n";
        send(sockfd, json_str.c_str(), json_str.length(), 0);

        char buffer[256];
        recv(sockfd, buffer, sizeof(buffer), 0);
        
        close(sockfd);
    } catch (...) {
        if (sockfd >= 0) close(sockfd);
    }
}

// Process horizontal swimming head contour (reuse existing algorithm)
int processHeadContour(const std::vector<std::pair<int, std::vector<double>>>& head_contour_points,
                       std::vector<std::pair<int, std::vector<double>>>& sampled_points,
                       size_t struct_id, int mpi_rank,
                       const std::vector<std::pair<int, std::vector<double>>>& fish_points,
                       bool head_at_larger_coord,
                       bool use_fallback_strategy) {
    
    bool used_extreme_point_method = false;
    if (head_contour_points.empty()) return 0;
    
    // Get fish body x range for centerline calculation
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    for (const auto& point : fish_points) {
        min_x = std::min(min_x, point.second[0]);
        max_x = std::max(max_x, point.second[0]);
    }
    
    // Sort fish body points by x coordinate
    std::vector<std::pair<int, std::vector<double>>> sorted_fish_points = fish_points;
    std::sort(sorted_fish_points.begin(), sorted_fish_points.end(), 
        [](const auto& a, const auto& b) { return a.second[0] < b.second[0]; });
    
    // Calculate centerline
    const int num_bins = 50;
    const double bin_width = (max_x - min_x) / num_bins;
    std::vector<double> median_y_values(num_bins, 0.0);
    std::vector<int> bin_counts(num_bins, 0);

    for (const auto& point : sorted_fish_points) {
        int bin_idx = std::min(num_bins-1, static_cast<int>((point.second[0] - min_x) / bin_width));
        median_y_values[bin_idx] += point.second[1];
        bin_counts[bin_idx]++;
    }

    for (int i = 0; i < num_bins; i++) {
        if (bin_counts[i] > 0) {
            median_y_values[i] /= bin_counts[i];
        } else if (i > 0 && i < num_bins - 1) {
            double sum = 0.0;
            int count = 0;
            if (bin_counts[i-1] > 0) { sum += median_y_values[i-1]; count++; }
            if (bin_counts[i+1] > 0) { sum += median_y_values[i+1]; count++; }
            median_y_values[i] = (count > 0) ? sum / count : 0.0;
        }
    }

    // Smooth centerline
    std::vector<double> smoothed_median(num_bins);
    for (int i = 0; i < num_bins; i++) {
        if (i == 0) {
            smoothed_median[i] = median_y_values[i];
        } else if (i == num_bins - 1) {
            smoothed_median[i] = median_y_values[i];
        } else {
            smoothed_median[i] = (median_y_values[i-1] + median_y_values[i] + median_y_values[i+1]) / 3.0;
        }
    }

    // Divide into upper and lower halves
    std::vector<std::pair<int, std::vector<double>>> upper_half;
    std::vector<std::pair<int, std::vector<double>>> lower_half;
    
    for (const auto& point : sorted_fish_points) {
        int bin_idx = std::min(num_bins-1, static_cast<int>((point.second[0] - min_x) / bin_width));
        double median_y = smoothed_median[bin_idx];
        
        if (point.second[1] >= median_y) {
            upper_half.push_back(point);
        } else {
            lower_half.push_back(point);
        }
    }

    // Sort upper and lower halves by x coordinate
    std::sort(upper_half.begin(), upper_half.end(), 
        [](const auto& a, const auto& b) { return a.second[0] < b.second[0]; });
    std::sort(lower_half.begin(), lower_half.end(), 
        [](const auto& a, const auto& b) { return a.second[0] < b.second[0]; });
    
    // Extract upper and lower head contour points
    std::vector<std::pair<int, std::vector<double>>> upper_contour;
    std::vector<std::pair<int, std::vector<double>>> lower_contour;
    
    for (const auto& point : head_contour_points) {
        int bin_idx = std::min(num_bins-1, static_cast<int>((point.second[0] - min_x) / bin_width));
        double median_y = smoothed_median[bin_idx];
        
        if (point.second[1] >= median_y) {
            upper_contour.push_back(point);
        } else {
            lower_contour.push_back(point);
        }
    }

    if (upper_contour.empty() || lower_contour.empty()) {
        pout << "[Process " << mpi_rank << "] Fish " << struct_id 
            << " insufficient head contour points, upper: " << upper_contour.size() 
            << ", lower: " << lower_contour.size() << "\n";
        return 0;
    }

     // Sort upper and lower parts by x coordinate
     std::sort(upper_contour.begin(), upper_contour.end(),
     [](const auto& a, const auto& b) { return a.second[0] < b.second[0]; });
 std::sort(lower_contour.begin(), lower_contour.end(),
     [](const auto& a, const auto& b) { return a.second[0] < b.second[0]; });
 
 // Create contour point sequence
 std::vector<std::pair<int, std::vector<double>>> ordered_contour_points;
 
 if (head_at_larger_coord) {
     // Head on right, create counterclockwise contour
     // Path: lower-left -> lower-right -> upper-right -> upper-left
     for (size_t i = 0; i < lower_contour.size(); ++i) {
         ordered_contour_points.push_back(lower_contour[i]);
     }
     for (int i = upper_contour.size() - 1; i >= 0; --i) {
         ordered_contour_points.push_back(upper_contour[i]);
     }
 } else {
     // Head on left, create true counterclockwise contour  
     // Corrected path: upper-right -> upper-left -> lower-left -> lower-right
     for (int i = upper_contour.size() - 1; i >= 0; --i) {
         ordered_contour_points.push_back(upper_contour[i]);
     }
     for (size_t i = 0; i < lower_contour.size(); ++i) {
         ordered_contour_points.push_back(lower_contour[i]);
     }
 }
 
 // Reorder contour points based on distance
 std::vector<std::pair<int, std::vector<double>>> reordered_points;
 std::vector<std::pair<int, std::vector<double>>> remaining_points = ordered_contour_points;
 
 if (!remaining_points.empty()) {
     auto start_point = remaining_points[0];
     reordered_points.push_back(start_point);
     remaining_points.erase(remaining_points.begin());
     
     auto current_point = start_point;
     while (!remaining_points.empty()) {
         std::vector<double> distances;
         for (const auto& point : remaining_points) {
             double dx = point.second[0] - current_point.second[0];
             double dy = point.second[1] - current_point.second[1];
             double dist = std::sqrt(dx*dx + dy*dy);
             distances.push_back(dist);
         }
         
         auto min_iter = std::min_element(distances.begin(), distances.end());
         size_t nearest_idx = std::distance(distances.begin(), min_iter);
         auto nearest_point = remaining_points[nearest_idx];
         
         reordered_points.push_back(nearest_point);
         current_point = nearest_point;
         remaining_points.erase(remaining_points.begin() + nearest_idx);
     }
     
     ordered_contour_points = reordered_points;
 }

 std::vector<std::vector<int>> all_straight_segments;

 // Calculate angles and find straight segments
 if (!ordered_contour_points.empty()) {
     std::vector<double> angles;
     for (size_t j = 0; j < ordered_contour_points.size() - 1; ++j) {
         double dx = ordered_contour_points[j+1].second[0] - ordered_contour_points[j].second[0];
         double dy = ordered_contour_points[j+1].second[1] - ordered_contour_points[j].second[1];
         double angle = std::atan2(dy, dx);  // Calculate angle, range [-π, π]
         angles.push_back(angle);
     }
     
     std::vector<double> angle_diffs;
     for (size_t i = 0; i < angles.size() - 1; ++i) {
         double diff = std::abs(angles[i+1] - angles[i]);
         // Handle angle crossing ±π boundary
         if (diff > M_PI) {
             diff = 2.0 * M_PI - diff;
         }
         angle_diffs.push_back(diff);
     }
     
     // New head identification logic
     std::vector<int> head_points;
     std::vector<int> current_segment;

     for (size_t j = 0; j < angle_diffs.size(); ++j) {
         if (angle_diffs[j] < 0.02) {
             if (current_segment.empty()) {
                 current_segment = {static_cast<int>(j), static_cast<int>(j)};
             } else {
                 current_segment[1] = static_cast<int>(j);
             }
         } else {
             if (!current_segment.empty()) {
                 all_straight_segments.push_back({current_segment[0], current_segment[1] + 2});
             }
             current_segment.clear();
         }
     }
     
     if (!current_segment.empty()) {
         all_straight_segments.push_back({current_segment[0], current_segment[1] + 2});
     }
     
     if (all_straight_segments.size() == 5) {
          pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - Found 5 straight segments, selecting middle segment as head.\n";
          head_points = all_straight_segments[2]; // Select middle segment
     } 
     else if (use_fallback_strategy) { // Only execute when fallback strategy is enabled
         pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - Warning: Not found 5 straight segments (found " << all_straight_segments.size() << "). Will try new strategy.\n";
         
         // 1. Find extreme point in swimming direction
         auto extreme_iter = head_at_larger_coord ? 
             std::max_element(ordered_contour_points.begin(), ordered_contour_points.end(),
                 [](const auto& a, const auto& b) { return a.second[0] < b.second[0]; }) :
             std::min_element(ordered_contour_points.begin(), ordered_contour_points.end(),
                 [](const auto& a, const auto& b) { return a.second[0] < b.second[0]; });
         size_t extreme_idx = std::distance(ordered_contour_points.begin(), extreme_iter);

         // 2. Check if extreme point is on any straight segment, find the longest one
         std::vector<int> best_segment;
         int max_len = -1;

         for (const auto& segment : all_straight_segments) {
             int start = segment[0];
             int end = segment[1];
             if (static_cast<int>(extreme_idx) >= start && static_cast<int>(extreme_idx) <= end) { // Check if point is within segment (inclusive)
                 int len = end - start;
                 if (len > max_len) {
                     max_len = len;
                     best_segment = segment;
                 }
             }
         }
         
         // 3. If found longest straight segment containing extreme point, use it
         if (!best_segment.empty()) {
             pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - New strategy successful: Found longest straight segment containing extreme point.\n";
             head_points = best_segment;
         } else {
         // 4. If not found, fall back to original extreme point algorithm
             used_extreme_point_method = true;
             pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - New strategy failed: Extreme point not on any straight segment. Will fall back to extreme point algorithm.\n";
             int start_idx = std::max(0, static_cast<int>(extreme_idx) - 3);
             int end_idx = std::min(static_cast<int>(ordered_contour_points.size()) - 1, static_cast<int>(extreme_idx) + 4);
             head_points = {start_idx, end_idx};
         }
     }
     
     std::vector<std::pair<int, std::vector<double>>> final_head_contour_points;
     if (!head_points.empty()) {
         for (int i = head_points[0]; i <= head_points[1]; ++i) {
             if (i >= 0 && i < static_cast<int>(ordered_contour_points.size())) {
                 final_head_contour_points.push_back(ordered_contour_points[i]);
             }
         }
     }

     // Debug output: show selected head points
     pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - Debug info: Original best head point coordinates (total " << final_head_contour_points.size() << "):\n";
     for(const auto& p : final_head_contour_points) {
         pout << "  ( " << p.second[0] << ", " << p.second[1] << " )\n";
     }

     // Force head points to be 8
     if (!final_head_contour_points.empty()) {
         while (final_head_contour_points.size() < 8) {
             final_head_contour_points.push_back(final_head_contour_points.back());
         }
         if (final_head_contour_points.size() > 8) {
             final_head_contour_points.resize(8);
         }
     }
     
     pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - Finally found " << final_head_contour_points.size() << " head points.\n";

     // Add body sampling points
     std::vector<std::pair<int, std::vector<double>>> additional_sample_points;
     if (!final_head_contour_points.empty()) {
         // Logic modification (V4 - solve duplicate point problem)
         bool physical_right_is_lower_half = head_at_larger_coord;
         
         std::vector<std::pair<int, std::vector<double>>>& right_side_points = physical_right_is_lower_half ? lower_half : upper_half;
         std::vector<std::pair<int, std::vector<double>>>& left_side_points = physical_right_is_lower_half ? upper_half : lower_half;

         // Sample physical right side
         if (!right_side_points.empty()) {
             int points_collected = 0;
             if (head_at_larger_coord) { // Head on right, sample from back to front (skip first point)
                 for (int i = right_side_points.size() - 1 - 8; i >= 0 && points_collected < 6; i -= 15) {
                     additional_sample_points.push_back(right_side_points[i]);
                     points_collected++;
                 }
             } else { // Head on left, sample from front to back (skip first point)
                 for (size_t i = 8; i < right_side_points.size() && points_collected < 6; i += 15) {
                     additional_sample_points.push_back(right_side_points[i]);
                     points_collected++;
                 }
             }
         }

         // Sample physical left side
         if (!left_side_points.empty()) {
             int points_collected = 0;
             if (head_at_larger_coord) { // Head on right, sample from back to front (skip first point)
                  for (int i = left_side_points.size() - 1 - 8; i >= 0 && points_collected < 6; i -= 15) {
                     additional_sample_points.push_back(left_side_points[i]);
                     points_collected++;
                 }
             } else { // Head on left, sample from front to back (skip first point)
                 for (size_t i = 8; i < left_side_points.size() && points_collected < 6; i += 15) {
                     additional_sample_points.push_back(left_side_points[i]);
                     points_collected++;
                 }
             }
         }
     }
     pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - Found " << additional_sample_points.size() << " body sampling points.\n";

     // Merge sampling points
     sampled_points = final_head_contour_points;
     sampled_points.insert(sampled_points.end(), additional_sample_points.begin(), additional_sample_points.end());
     pout << "[Process " << mpi_rank << "] Fish " << struct_id << " horizontal swimming - Total sampling points: " << sampled_points.size() << ".\n";
 }
 return all_straight_segments.size(); // Return number of segments found
}

// Process vertical swimming head contour (change x coordinate logic to y coordinate)
int processHeadContourVertical(const std::vector<std::pair<int, std::vector<double>>>& head_contour_points,
                            std::vector<std::pair<int, std::vector<double>>>& sampled_points,
                            size_t struct_id, int mpi_rank,
                            const std::vector<std::pair<int, std::vector<double>>>& fish_points,
                            bool head_at_larger_coord,
                            bool use_fallback_strategy) {
 
 bool used_extreme_point_method = false;
 if (head_contour_points.empty()) return 0;
 
 // Get fish body y range for centerline calculation
 double min_y = std::numeric_limits<double>::max();
 double max_y = std::numeric_limits<double>::lowest();
 for (const auto& point : fish_points) {
     min_y = std::min(min_y, point.second[1]);
     max_y = std::max(max_y, point.second[1]);
 }
 
 // Sort fish body points by y coordinate
 std::vector<std::pair<int, std::vector<double>>> sorted_fish_points = fish_points;
 std::sort(sorted_fish_points.begin(), sorted_fish_points.end(), 
     [](const auto& a, const auto& b) { return a.second[1] < b.second[1]; });
 
 // Calculate centerline (swap x/y)
 const int num_bins = 50;
 const double bin_width = (max_y - min_y) / num_bins;
 std::vector<double> median_x_values(num_bins, 0.0);
 std::vector<int> bin_counts(num_bins, 0);

 for (const auto& point : sorted_fish_points) {
     int bin_idx = std::min(num_bins-1, static_cast<int>((point.second[1] - min_y) / bin_width));
     median_x_values[bin_idx] += point.second[0];
     bin_counts[bin_idx]++;
 }

 for (int i = 0; i < num_bins; i++) {
     if (bin_counts[i] > 0) {
         median_x_values[i] /= bin_counts[i];
     } else if (i > 0 && i < num_bins - 1) {
         double sum = 0.0;
         int count = 0;
         if (bin_counts[i-1] > 0) { sum += median_x_values[i-1]; count++; }
         if (bin_counts[i+1] > 0) { sum += median_x_values[i+1]; count++; }
         median_x_values[i] = (count > 0) ? sum / count : 0.0;
     }
 }

 // Smooth centerline
 std::vector<double> smoothed_median(num_bins);
 for (int i = 0; i < num_bins; i++) {
     if (i == 0) {
         smoothed_median[i] = median_x_values[i];
     } else if (i == num_bins - 1) {
         smoothed_median[i] = median_x_values[i];
     } else {
         smoothed_median[i] = (median_x_values[i-1] + median_x_values[i] + median_x_values[i+1]) / 3.0;
     }
 }

 // Divide into right and left halves (change upper/lower to right/left)
 std::vector<std::pair<int, std::vector<double>>> right_half;
 std::vector<std::pair<int, std::vector<double>>> left_half;
 
 for (const auto& point : sorted_fish_points) {
     int bin_idx = std::min(num_bins-1, static_cast<int>((point.second[1] - min_y) / bin_width));
     double median_x = smoothed_median[bin_idx];
     
     if (point.second[0] >= median_x) {
         right_half.push_back(point);
     } else {
         left_half.push_back(point);
     }
 }

 // Sort right and left halves by y coordinate
 std::sort(right_half.begin(), right_half.end(), 
     [](const auto& a, const auto& b) { return a.second[1] < b.second[1]; });
 std::sort(left_half.begin(), left_half.end(), 
     [](const auto& a, const auto& b) { return a.second[1] < b.second[1]; });
 
 // Extract right and left head contour points
 std::vector<std::pair<int, std::vector<double>>> right_contour;
 std::vector<std::pair<int, std::vector<double>>> left_contour;
 
 for (const auto& point : head_contour_points) {
     int bin_idx = std::min(num_bins-1, static_cast<int>((point.second[1] - min_y) / bin_width));
     double median_x = smoothed_median[bin_idx];
     
     if (point.second[0] >= median_x) {
         right_contour.push_back(point);
     } else {
         left_contour.push_back(point);
     }
 }

 if (right_contour.empty() || left_contour.empty()) {
     pout << "[Process " << mpi_rank << "] Fish " << struct_id 
         << " insufficient head contour points, right: " << right_contour.size() 
         << ", left: " << left_contour.size() << "\n";
     return 0;
 }

 // Sort right and left parts by y coordinate
 std::sort(right_contour.begin(), right_contour.end(),
     [](const auto& a, const auto& b) { return a.second[1] < b.second[1]; });
 std::sort(left_contour.begin(), left_contour.end(),
     [](const auto& a, const auto& b) { return a.second[1] < b.second[1]; });
 
 // Create contour point sequence
 std::vector<std::pair<int, std::vector<double>>> ordered_contour_points;
 
 if (head_at_larger_coord) {
     // Head on top, create counterclockwise contour consistent with "head on bottom" case
     // Corrected order: lower-right -> upper-right -> upper-left -> lower-left
     for (size_t i = 0; i < right_contour.size(); ++i) {
         ordered_contour_points.push_back(right_contour[i]);
     }
     for (int i = left_contour.size() - 1; i >= 0; --i) {
         ordered_contour_points.push_back(left_contour[i]);
     }
 } else {
     // Head on bottom, create contour from bottom to top (upper-left -> lower-left -> lower-right -> upper-right, maintain counterclockwise)
     for (int i = left_contour.size() - 1; i >= 0; --i) {
         ordered_contour_points.push_back(left_contour[i]);
     }
     for (size_t i = 0; i < right_contour.size(); ++i) {
         ordered_contour_points.push_back(right_contour[i]);
     }
 }
 
 // Reorder contour points based on distance (same as horizontal version)
 std::vector<std::pair<int, std::vector<double>>> reordered_points;
 std::vector<std::pair<int, std::vector<double>>> remaining_points = ordered_contour_points;
 
 if (!remaining_points.empty()) {
     auto start_point = remaining_points[0];
     reordered_points.push_back(start_point);
     remaining_points.erase(remaining_points.begin());
     
     auto current_point = start_point;
     while (!remaining_points.empty()) {
         std::vector<double> distances;
         for (const auto& point : remaining_points) {
             double dx = point.second[0] - current_point.second[0];
             double dy = point.second[1] - current_point.second[1];
             double dist = std::sqrt(dx*dx + dy*dy);
             distances.push_back(dist);
         }
         
         auto min_iter = std::min_element(distances.begin(), distances.end());
         size_t nearest_idx = std::distance(distances.begin(), min_iter);
         auto nearest_point = remaining_points[nearest_idx];
         
         reordered_points.push_back(nearest_point);
         current_point = nearest_point;
         remaining_points.erase(remaining_points.begin() + nearest_idx);
     }
     
     ordered_contour_points = reordered_points;
 }

 std::vector<std::vector<int>> all_straight_segments;

  if (!ordered_contour_points.empty()) {
     std::vector<double> angles;
     for (size_t j = 0; j < ordered_contour_points.size() - 1; ++j) {
         double dx = ordered_contour_points[j+1].second[0] - ordered_contour_points[j].second[0];
         double dy = ordered_contour_points[j+1].second[1] - ordered_contour_points[j].second[1];
         angles.push_back(atan2(dy, dx));
     }
     
     std::vector<double> angle_diffs;
     for (size_t i = 0; i < angles.size() - 1; ++i) {
         double diff = angles[i+1] - angles[i];
         // Handle angle jump from pi to -pi (or vice versa)
         if (diff > M_PI) diff -= 2 * M_PI;
         if (diff <= -M_PI) diff += 2 * M_PI;
         angle_diffs.push_back(std::abs(diff));
     }
     
     // New head identification logic
     std::vector<int> head_points;
     std::vector<int> current_segment;

     for (size_t j = 0; j < angle_diffs.size(); ++j) {
         if (angle_diffs[j] < 0.02) { 
             if (current_segment.empty()) {
                 current_segment = {static_cast<int>(j), static_cast<int>(j)};
             } else {
                 current_segment[1] = static_cast<int>(j);
             }
         } else {
             if (!current_segment.empty()) {
                  all_straight_segments.push_back({current_segment[0], current_segment[1] + 2});
             }
             current_segment.clear();
         }
     }
     
     if (!current_segment.empty()) {
          all_straight_segments.push_back({current_segment[0], current_segment[1] + 2});
     }
     
     if (all_straight_segments.size() == 5) {
         pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - Found 5 straight segments, selecting middle segment as head.\n";
         head_points = all_straight_segments[2]; // Select middle segment
     }
     else if (use_fallback_strategy) { // Only execute when fallback strategy is enabled
         pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - Warning: Not found 5 straight segments (found " << all_straight_segments.size() << "). Will try new strategy.\n";
         
         // 1. Find extreme point in swimming direction
         auto extreme_iter = head_at_larger_coord ? 
             std::max_element(ordered_contour_points.begin(), ordered_contour_points.end(),
                 [](const auto& a, const auto& b) { return a.second[1] < b.second[1]; }) :
             std::min_element(ordered_contour_points.begin(), ordered_contour_points.end(),
                 [](const auto& a, const auto& b) { return a.second[1] < b.second[1]; });
         size_t extreme_idx = std::distance(ordered_contour_points.begin(), extreme_iter);

         // 2. Check if extreme point is on any straight segment, find the longest one
         std::vector<int> best_segment;
         int max_len = -1;

         for (const auto& segment : all_straight_segments) {
             int start = segment[0];
             int end = segment[1];
             if (static_cast<int>(extreme_idx) >= start && static_cast<int>(extreme_idx) <= end) { // Check if point is within segment (inclusive)
                 int len = end - start;
                 if (len > max_len) {
                     max_len = len;
                     best_segment = segment;
                 }
             }
         }
         
         // 3. If found longest straight segment containing extreme point, use it
         if (!best_segment.empty()) {
             pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - New strategy successful: Found longest straight segment containing extreme point.\n";
             head_points = best_segment;
         } else {
         // 4. If not found, fall back to original extreme point algorithm
             used_extreme_point_method = true;
             pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - New strategy failed: Extreme point not on any straight segment. Will fall back to extreme point algorithm.\n";
             int start_idx = std::max(0, static_cast<int>(extreme_idx) - 3);
             int end_idx = std::min(static_cast<int>(ordered_contour_points.size()) - 1, static_cast<int>(extreme_idx) + 4);
             head_points = {start_idx, end_idx};
         }
     }
     
     std::vector<std::pair<int, std::vector<double>>> final_head_contour_points;
     if (!head_points.empty()) {
         for (int i = head_points[0]; i <= head_points[1]; ++i) {
             if (i >= 0 && i < static_cast<int>(ordered_contour_points.size())) {
                 final_head_contour_points.push_back(ordered_contour_points[i]);
             }
         }
     }

     // Debug output: show selected head points
     pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - Debug info: Original best head point coordinates (total " << final_head_contour_points.size() << "):\n";
     for(const auto& p : final_head_contour_points) {
         pout << "  ( " << p.second[0] << ", " << p.second[1] << " )\n";
     }

     // Force head points to be 8
     if (!final_head_contour_points.empty()) {
          while (final_head_contour_points.size() < 8) {
             final_head_contour_points.push_back(final_head_contour_points.back());
         }
         if (final_head_contour_points.size() > 8) {
             final_head_contour_points.resize(8);
         }
     }
     
     pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - Finally found " << final_head_contour_points.size() << " head points.\n";

     // Add body sampling points (along y direction)
     std::vector<std::pair<int, std::vector<double>>> additional_sample_points;
     if (!final_head_contour_points.empty()) {
         // Logic modification (V4 - solve duplicate point problem)
         bool physical_right_is_right_half = head_at_larger_coord;

         std::vector<std::pair<int, std::vector<double>>>& right_side_points = physical_right_is_right_half ? right_half : left_half;
         std::vector<std::pair<int, std::vector<double>>>& left_side_points = physical_right_is_right_half ? left_half : right_half;

         // Sample physical right side
         if (!right_side_points.empty()) {
             int points_collected = 0;
             if (head_at_larger_coord) { // Head on top, sample from back to front (skip first point)
                 for (int i = right_side_points.size() - 1 - 8; i >= 0 && points_collected < 6; i -= 15) {
                     additional_sample_points.push_back(right_side_points[i]);
                     points_collected++;
                 }
             } else { // Head on bottom, sample from front to back (skip first point)
                 for (size_t i = 8; i < right_side_points.size() && points_collected < 6; i += 15) {
                     additional_sample_points.push_back(right_side_points[i]);
                     points_collected++;
                 }
             }
         }

         // Sample physical left side
         if (!left_side_points.empty()) {
             int points_collected = 0;
              if (head_at_larger_coord) { // Head on top, sample from back to front (skip first point)
                 for (int i = left_side_points.size() - 1 - 8; i >= 0 && points_collected < 6; i -= 15) {
                     additional_sample_points.push_back(left_side_points[i]);
                     points_collected++;
                 }
             } else { // Head on bottom, sample from front to back (skip first point)
                 for (size_t i = 8; i < left_side_points.size() && points_collected < 6; i += 15) {
                     additional_sample_points.push_back(left_side_points[i]);
                     points_collected++;
                 }
             }
         }
     }
     pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - Found " << additional_sample_points.size() << " body sampling points.\n";

     // Merge sampling points
     sampled_points = final_head_contour_points;
     sampled_points.insert(sampled_points.end(), additional_sample_points.begin(), additional_sample_points.end());
     pout << "[Process " << mpi_rank << "] Fish " << struct_id << " vertical swimming - Total sampling points: " << sampled_points.size() << ".\n";
 }
 return all_straight_segments.size(); // Return number of segments found
}