// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2022 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

//////////////////////////// INCLUDES /////////////////////////////////////////
#include "ibtk/IBTK_MPI.h"

#include "CartesianPatchGeometry.h"
#include "IBEELKinematics.h"
#include "PatchLevel.h"
#include "tbox/MathUtilities.h"

#include "muParser.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include "ibamr/namespaces.h"

namespace IBAMR
{
namespace
{
inline int
sign(const double X)
{
    return ((X > 0) ? 1 : ((X < 0) ? -1 : 0));
}

static const double PII = 3.1415926535897932384626433832795;
static const double __INFINITY = 1e9;

// Fish geometry parameters
static const double LENGTH_FISH = 1.0;
static const double WIDTH_HEAD = 0.04 * LENGTH_FISH;
static const double LENGTH_HEAD = 0.04;

// Prey capture parameters
static const double CUT_OFF_ANGLE = PII / 4;
static const double CUT_OFF_RADIUS = 0.7;
static const double LOWER_CUT_OFF_ANGLE = 7 * PII / 180;

} // namespace

///////////////////////////////////////////////////////////////////////

IBEELKinematics::IBEELKinematics(const std::string& object_name,
                                 Pointer<Database> input_db,
                                 LDataManager* l_data_manager,
                                 Pointer<PatchHierarchy<NDIM> > patch_hierarchy,
                                 bool register_for_restart)
    : ConstraintIBKinematics(object_name, input_db, l_data_manager, register_for_restart),
      d_current_time(0.0),
      d_kinematics_vel(NDIM),
      d_shape(NDIM),
      d_center_of_mass(3),
      d_incremented_angle_from_reference_axis(3),
      d_tagged_pt_position(3),
      d_mesh_width(NDIM),
      d_parser_time(0.0),
      d_c0(0.0),
      d_c1(0.0),
      d_c2(0.0),
      d_c3(0.0),
      d_c4(0.0),
      d_c5(0.0),
      d_integration_step_size(0.001),  // Integration step size
      d_use_exact_integration(true),   // Use exact integration by default
      d_time_step_size(0.001),         // Time step size
      d_wave_function_lambda(1.0),
      d_wave_function_T_n(1.0),
      d_cache_initialized(false),
      d_time_offset(0.0)  // Initialize time offset to 0
{
    // Read from input database
    d_initAngle_bodyAxis_x = input_db->getDoubleWithDefault("initial_angle_body_axis_0", 0.0);
    d_bodyIsManeuvering = input_db->getBoolWithDefault("body_is_maneuvering", false);
    d_maneuverAxisIsChangingShape = input_db->getBoolWithDefault("maneuvering_axis_is_changing_shape", false);

    // Read deformation velocity functions
    std::vector<std::string> deformationvel_function_strings;
    for (int d = 0; d < NDIM; ++d)
    {
        const std::string postfix = "_function_" + std::to_string(d);
        std::string key_name = "deformation_velocity" + postfix;

        if (input_db->isString(key_name))
        {
            deformationvel_function_strings.push_back(input_db->getString(key_name));
        }
        else
        {
            deformationvel_function_strings.push_back("0.0");
            TBOX_WARNING("IBEELKinematics::IBEELKinematics() :\n"
                         << "  no function corresponding to key ``" << key_name << " '' found for dimension = " << d
                         << "; using def_vel = 0.0. " << std::endl);
        }

        d_deformationvel_parsers.push_back(new mu::Parser());
        d_deformationvel_parsers.back()->SetExpr(deformationvel_function_strings.back());
        d_all_parsers.push_back(d_deformationvel_parsers.back());
    }

    // Read body shape parser
    {
        const std::string body_shape_equation = input_db->getString("body_shape_equation");
        d_body_shape_parser = new mu::Parser();
        d_body_shape_parser->SetExpr(body_shape_equation);
        d_all_parsers.push_back(d_body_shape_parser);
    }

    // Read maneuvering axis parser
    if (d_bodyIsManeuvering)
    {
        const std::string maneuvering_axis_equation = input_db->getString("maneuvering_axis_equation");
        d_maneuvering_axis_parser = new mu::Parser();
        d_maneuvering_axis_parser->SetExpr(maneuvering_axis_equation);
        d_all_parsers.push_back(d_maneuvering_axis_parser);
    }

    // Define constants for parsers
    const double pi = 3.1415926535897932384626433832795;
    for (std::vector<mu::Parser*>::const_iterator cit = d_all_parsers.begin(); cit != d_all_parsers.end(); ++cit)
    {
        // Pi constants
        (*cit)->DefineConst("pi", pi);
        (*cit)->DefineConst("Pi", pi);
        (*cit)->DefineConst("PI", pi);

        // Variables
        (*cit)->DefineVar("T", &d_parser_time);
        (*cit)->DefineVar("t", &d_parser_time);
        for (int d = 0; d < NDIM; ++d)
        {
            const std::string postfix = std::to_string(d);
            (*cit)->DefineVar("X" + postfix, d_parser_posn.data() + d);
            (*cit)->DefineVar("x" + postfix, d_parser_posn.data() + d);
            (*cit)->DefineVar("X_" + postfix, d_parser_posn.data() + d);
            (*cit)->DefineVar("x_" + postfix, d_parser_posn.data() + d);

            (*cit)->DefineVar("N" + postfix, d_parser_normal.data() + d);
            (*cit)->DefineVar("n" + postfix, d_parser_normal.data() + d);
            (*cit)->DefineVar("N_" + postfix, d_parser_normal.data() + d);
            (*cit)->DefineVar("n_" + postfix, d_parser_normal.data() + d);
        }

        // Register coefficient constants
        (*cit)->DefineVar("c0", &d_c0);
        (*cit)->DefineVar("c1", &d_c1);
        (*cit)->DefineVar("c2", &d_c2);
        (*cit)->DefineVar("c3", &d_c3);
        (*cit)->DefineVar("c4", &d_c4);
        (*cit)->DefineVar("c5", &d_c5);
    }

    // Set food particle location from input file
    d_food_location.resizeArray(NDIM);
    for (int dim = 0; dim < NDIM; ++dim)
    {
        d_food_location[dim] = input_db->getDouble("food_location_in_domain_" + std::to_string(dim));
    }

    // Set immersed body layout in reference frame
    setImmersedBodyLayout(patch_hierarchy);

    // Read integration parameters from input database
    d_integration_step_size = input_db->getDoubleWithDefault("integration_step_size", 0.001);
    d_use_exact_integration = input_db->getBoolWithDefault("use_exact_integration", true);
    d_time_step_size = input_db->getDoubleWithDefault("time_step_size", 0.001);
    d_wave_function_lambda = input_db->getDoubleWithDefault("wave_function_lambda", 1.0);
    d_wave_function_T_n = input_db->getDoubleWithDefault("wave_function_T_n", 1.0);

    // Read time offset from input file
    d_time_offset = input_db->getDoubleWithDefault("LAST_COEFF_UPDATE_TIME", 0.0);

    // Output read values on main process for confirmation
    if (IBTK_MPI::getRank() == 0) {
        pout << "Time offset read from input file d_time_offset = " << d_time_offset << std::endl;
    }

    bool from_restart = RestartManager::getManager()->isFromRestart();
    if (from_restart) getFromRestart();

    return;

} // IBEELKinematics

IBEELKinematics::~IBEELKinematics()
{
    for (std::vector<mu::Parser*>::const_iterator cit = d_all_parsers.begin(); cit != d_all_parsers.end(); ++cit)
    {
        delete (*cit);
    }
    return;

} // ~IBEELKinematics

void
IBEELKinematics::putToDatabase(Pointer<Database> db)
{
    db->putDouble("d_current_time", d_current_time);
    db->putDouble("d_time_offset", d_time_offset);
    db->putDoubleArray("d_center_of_mass", &d_center_of_mass[0], 3);
    db->putDoubleArray("d_incremented_angle_from_reference_axis", &d_incremented_angle_from_reference_axis[0], 3);
    db->putDoubleArray("d_tagged_pt_position", &d_tagged_pt_position[0], 3);

    // Save waveform function parameters
    db->putDouble("d_wave_function_lambda", d_wave_function_lambda);
    db->putDouble("d_wave_function_T_n", d_wave_function_T_n);
    db->putBool("d_cache_initialized", d_cache_initialized);
    
    // Save coefficient cache
    if (d_cache_initialized && d_coefficient_cache.size() == 3) {
        for (int i = 0; i < 3; ++i) {
            std::string prefix = "cache_" + std::to_string(i) + "_";
            db->putDouble(prefix + "theta_lmax_prev", d_coefficient_cache[i].theta_lmax_prev);
            db->putDouble(prefix + "theta_lmax_next", d_coefficient_cache[i].theta_lmax_next);
            db->putDouble(prefix + "lambda_prev", d_coefficient_cache[i].lambda_prev);
            db->putDouble(prefix + "lambda_next", d_coefficient_cache[i].lambda_next);
            db->putDouble(prefix + "c0", d_coefficient_cache[i].c0);
            db->putDouble(prefix + "c1", d_coefficient_cache[i].c1);
            db->putDouble(prefix + "c2", d_coefficient_cache[i].c2);
            db->putDouble(prefix + "c3", d_coefficient_cache[i].c3);
            db->putDouble(prefix + "c4", d_coefficient_cache[i].c4);
            db->putDouble(prefix + "c5", d_coefficient_cache[i].c5);
        }
    }

    return;
} // putToDatabase

void
IBEELKinematics::getFromRestart()
{
    Pointer<Database> restart_db = RestartManager::getManager()->getRootDatabase();
    Pointer<Database> db;
    if (restart_db->isDatabase(d_object_name))
    {
        db = restart_db->getDatabase(d_object_name);
    }
    else
    {
        TBOX_ERROR(d_object_name << ":  Restart database corresponding to " << d_object_name
                                 << " not found in restart file." << std::endl);
    }

    d_current_time = db->getDouble("d_current_time");
    d_time_offset = db->getDoubleWithDefault("d_time_offset", 0.0);
    
    // Print values read from restart file
    if (IBTK_MPI::getRank() == 0) {
        pout << "Time offset read from restart file d_time_offset = " << d_time_offset << std::endl;
        pout << "Current time read from restart file d_current_time = " << d_current_time << std::endl;
    }
    
    db->getDoubleArray("d_center_of_mass", &d_center_of_mass[0], 3);
    db->getDoubleArray("d_incremented_angle_from_reference_axis", &d_incremented_angle_from_reference_axis[0], 3);
    db->getDoubleArray("d_tagged_pt_position", &d_tagged_pt_position[0], 3);

    // Restore waveform function parameters
    d_wave_function_lambda = db->getDoubleWithDefault("d_wave_function_lambda", 1.0);
    d_wave_function_T_n = db->getDoubleWithDefault("d_wave_function_T_n", 1.0);

    if (IBTK_MPI::getRank() == 0) {
        pout << "Wave function lambda read from restart file d_wave_function_lambda = " << d_wave_function_lambda << std::endl;
        pout << "Wave function T_n read from restart file d_wave_function_T_n = " << d_wave_function_T_n << std::endl;
        pout << "Cache initialized read from restart file d_cache_initialized = " << (d_cache_initialized ? "true" : "false") << std::endl;
    }

    return;
} // getFromRestart

void
IBEELKinematics::setImmersedBodyLayout(Pointer<PatchHierarchy<NDIM> > patch_hierarchy)
{
    // Set vector sizes
    const StructureParameters& struct_param = getStructureParameters();
    const int coarsest_ln = struct_param.getCoarsestLevelNumber();
    const int finest_ln = struct_param.getFinestLevelNumber();
    TBOX_ASSERT(coarsest_ln == finest_ln);
    const std::vector<std::pair<int, int> >& idx_range = struct_param.getLagIdxRange();
    const int total_lag_pts = idx_range[0].second - idx_range[0].first;

    for (int d = 0; d < NDIM; ++d)
    {
        d_kinematics_vel[d].resize(total_lag_pts);
        d_shape[d].resize(total_lag_pts);
    }

    // Get background mesh data
    Pointer<PatchLevel<NDIM> > level = patch_hierarchy->getPatchLevel(finest_ln);
    PatchLevel<NDIM>::Iterator p(level);
    Pointer<Patch<NDIM> > patch = level->getPatch(p());
    Pointer<CartesianPatchGeometry<NDIM> > pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    for (int dim = 0; dim < NDIM; ++dim)
    {
        d_mesh_width[dim] = dx[dim];
    }

    // Number of points on backbone and head
    const int BodyNx = static_cast<int>(ceil(LENGTH_FISH / d_mesh_width[0]));
    const int HeadNx = static_cast<int>(ceil(LENGTH_HEAD / d_mesh_width[0]));

    d_ImmersedBodyData.clear();
    for (int i = 1; i <= HeadNx; ++i)
    {
        const double s = (i - 1) * d_mesh_width[0];
        const double section = sqrt(2 * WIDTH_HEAD * s - s * s);
        const int NumPtsInSection = 2 * static_cast<int>(ceil(section / d_mesh_width[1]));
        d_ImmersedBodyData.insert(std::make_pair(s, NumPtsInSection));
    }

    for (int i = HeadNx + 1; i <= BodyNx; ++i)
    {
        const double s = (i - 1) * d_mesh_width[0];
        const double section = WIDTH_HEAD * (LENGTH_FISH - s) / (LENGTH_FISH - LENGTH_HEAD);
        const int NumPtsInHeight = 2 * static_cast<int>(ceil(section / d_mesh_width[1]));
        d_ImmersedBodyData.insert(std::make_pair(s, NumPtsInHeight));
    }

    // Find maneuvering axis coordinates in reference frame
    if (d_bodyIsManeuvering)
    {
        d_maneuverAxisReferenceCoordinates_vec.clear();
        d_map_reference_tangent.clear();
        d_map_reference_sign.clear();

        std::vector<double> vec_axis_coord(2);
        for (std::map<double, int>::const_iterator mitr = d_ImmersedBodyData.begin(); mitr != d_ImmersedBodyData.end();
             ++mitr)
        {
            d_parser_posn[0] = mitr->first;
            vec_axis_coord[0] = mitr->first;
            vec_axis_coord[1] = d_maneuvering_axis_parser->Eval();
            d_maneuverAxisReferenceCoordinates_vec.push_back(vec_axis_coord);
        }

        // Store tangents to reference maneuver axis
        for (unsigned int i = 0; i <= (d_maneuverAxisReferenceCoordinates_vec.size() - 2); ++i)
        {
            std::vector<int> sign_vec(2);
            const double s = d_maneuverAxisReferenceCoordinates_vec[i][0];
            const double dX =
                (d_maneuverAxisReferenceCoordinates_vec[i + 1][0] - d_maneuverAxisReferenceCoordinates_vec[i][0]);
            const double dY =
                (d_maneuverAxisReferenceCoordinates_vec[i + 1][1] - d_maneuverAxisReferenceCoordinates_vec[i][1]);
            sign_vec[0] = sign(dX);
            sign_vec[1] = sign(dY);
            const double theta = std::atan(std::abs(dY / dX));
            d_map_reference_tangent.insert(std::make_pair(s, theta));
            d_map_reference_sign.insert(std::make_pair(s, sign_vec));
        }

        // Fill in last point in map
        d_map_reference_tangent.insert(std::make_pair((d_maneuverAxisReferenceCoordinates_vec.back())[0],
                                                      (d_map_reference_tangent.rbegin())->second));
        d_map_reference_sign.insert(std::make_pair((d_maneuverAxisReferenceCoordinates_vec.back())[0],
                                                   (d_map_reference_sign.rbegin())->second));

        // Find COM of maneuver axis
        double maneuverAxis_x_cm = 0.0;
        double maneuverAxis_y_cm = 0.0;
        for (unsigned int i = 0; i < d_maneuverAxisReferenceCoordinates_vec.size(); ++i)
        {
            maneuverAxis_x_cm += d_maneuverAxisReferenceCoordinates_vec[i][0];
            maneuverAxis_y_cm += d_maneuverAxisReferenceCoordinates_vec[i][1];
        }
        maneuverAxis_x_cm /= d_maneuverAxisReferenceCoordinates_vec.size();
        maneuverAxis_y_cm /= d_maneuverAxisReferenceCoordinates_vec.size();

        // Shift reference so maneuver axis COM coincides with origin
        for (unsigned int i = 0; i < d_maneuverAxisReferenceCoordinates_vec.size(); ++i)
        {
            d_maneuverAxisReferenceCoordinates_vec[i][0] -= maneuverAxis_x_cm;
            d_maneuverAxisReferenceCoordinates_vec[i][1] -= maneuverAxis_y_cm;
        }
    }

    return;

} // setImmersedBodyLayout

void
IBEELKinematics::transformManeuverAxisAndCalculateTangents(const double angleFromHorizontal)
{
    d_maneuverAxisTransformedCoordinates_vec.clear();
    d_map_transformed_tangent.clear();
    d_map_transformed_sign.clear();

    const int BodyNx = static_cast<int>(ceil(LENGTH_FISH / d_mesh_width[0]));
    std::vector<double> transformed_coord(2);
    for (int i = 0; i <= (BodyNx - 1); ++i)
    {
        transformed_coord[0] = d_maneuverAxisReferenceCoordinates_vec[i][0] * cos(angleFromHorizontal) -
                               d_maneuverAxisReferenceCoordinates_vec[i][1] * sin(angleFromHorizontal);
        transformed_coord[1] = d_maneuverAxisReferenceCoordinates_vec[i][0] * sin(angleFromHorizontal) +
                               d_maneuverAxisReferenceCoordinates_vec[i][1] * cos(angleFromHorizontal);
        d_maneuverAxisTransformedCoordinates_vec.push_back(transformed_coord);
    }

    for (int i = 0; i <= (BodyNx - 2); ++i)
    {
        std::vector<int> sign_vec(2);
        const double s = i * d_mesh_width[0];
        const double dX =
            (d_maneuverAxisTransformedCoordinates_vec[i + 1][0] - d_maneuverAxisTransformedCoordinates_vec[i][0]);
        const double dY =
            (d_maneuverAxisTransformedCoordinates_vec[i + 1][1] - d_maneuverAxisTransformedCoordinates_vec[i][1]);
        sign_vec[0] = sign(dX);
        sign_vec[1] = sign(dY);
        const double theta = std::atan(std::abs(dY / dX));
        d_map_transformed_tangent.insert(std::make_pair(s, theta));
        d_map_transformed_sign.insert(std::make_pair(s, sign_vec));
    }

    // Fill in last point
    d_map_transformed_tangent.insert(
        std::make_pair((BodyNx - 1) * d_mesh_width[0], (d_map_transformed_tangent.rbegin())->second));
    d_map_transformed_sign.insert(
        std::make_pair((BodyNx - 1) * d_mesh_width[0], (d_map_transformed_sign.rbegin())->second));

    return;

} // transformManeuverAxisAndCalculateTangents

void
IBEELKinematics::setShape(const double time, const std::vector<double>& incremented_angle_from_reference_axis)
{
    const StructureParameters& struct_param = getStructureParameters();
    const std::string position_update_method = struct_param.getPositionUpdateMethod();
    if (position_update_method == "CONSTRAINT_VELOCITY") return;

    // Find deformed shape and rotate about center of mass
    TBOX_ASSERT(d_new_time == time);
    d_parser_time = time;
    std::vector<double> shape_new(NDIM);

    int lag_idx = -1;
    int reference_axis_idx = -1;
    for (std::map<double, int>::const_iterator itr = d_ImmersedBodyData.begin(); itr != d_ImmersedBodyData.end(); itr++)
    {
        const int NumPtsInSection = itr->second;
        const double s_l = itr->first;  // Arc length coordinate (from head end)
        
        const double y_shape_base = calculateHl(s_l, time); // Midline y-coordinate

        if (d_bodyIsManeuvering)
        {
            ++reference_axis_idx;
            const double x_maneuver_base = d_maneuverAxisReferenceCoordinates_vec[reference_axis_idx][0];
            const double y_maneuver_base = d_maneuverAxisReferenceCoordinates_vec[reference_axis_idx][1];

            for (int j = 1; j <= NumPtsInSection / 2; ++j)
            {
                const double nx = (-1 * sin(d_map_reference_tangent[itr->first]) * d_map_reference_sign[itr->first][1]);
                const double ny = (cos(d_map_reference_tangent[itr->first]) * d_map_reference_sign[itr->first][0]);

                // Apply intrinsic body deformation along normal to maneuver axis
                shape_new[0] = x_maneuver_base + (y_shape_base + (j - 1) * d_mesh_width[1]) * nx;
                shape_new[1] = y_maneuver_base + (y_shape_base + (j - 1) * d_mesh_width[1]) * ny;

                d_shape[0][++lag_idx] = shape_new[0];
                d_shape[1][lag_idx] = shape_new[1];
            }

            for (int j = 1; j <= NumPtsInSection / 2; ++j)
            {
                const double nx = (-1 * sin(d_map_reference_tangent[itr->first]) * d_map_reference_sign[itr->first][1]);
                const double ny = (cos(d_map_reference_tangent[itr->first]) * d_map_reference_sign[itr->first][0]);

                shape_new[0] = x_maneuver_base + (y_shape_base - (j)*d_mesh_width[1]) * nx;
                shape_new[1] = y_maneuver_base + (y_shape_base - (j)*d_mesh_width[1]) * ny;

                d_shape[0][++lag_idx] = shape_new[0];
                d_shape[1][lag_idx] = shape_new[1];
            }
        }
        else // Not maneuvering
        {
            // Calculate x-coordinate of midline by integrating cos(theta_l)
            const double x_shape_base = calculateXl(s_l, time);
            const double y_shape_base = calculateHl(s_l, time); // Midline y-coordinate

            for (int j = 1; j <= NumPtsInSection / 2; ++j) // Upper half of cross-section
            {
                // Use normal to define shape (consistent with Python script)
                double theta_l = calculateThetaL(s_l, time);
                double nx = -sin(theta_l);
                double ny = cos(theta_l);
                double dist = (j - 1) * d_mesh_width[1];
                d_shape[0][++lag_idx] = x_shape_base + dist * nx;
                d_shape[1][lag_idx] = y_shape_base + dist * ny;
            }

            for (int j = 1; j <= NumPtsInSection / 2; ++j) // Lower half of cross-section
            {
                double theta_l = calculateThetaL(s_l, time);
                double nx = -sin(theta_l);
                double ny = cos(theta_l);
                double dist = -j * d_mesh_width[1];
                d_shape[0][++lag_idx] = x_shape_base + dist * nx;
                d_shape[1][lag_idx] = y_shape_base + dist * ny;
            }
        }
    }

    // Find COM of new shape
    std::vector<double> center_of_mass(NDIM, 0.0);
    const int total_lag_pts = d_shape[0].size();
    for (int d = 0; d < NDIM; ++d)
    {
        for (std::vector<double>::const_iterator citr = d_shape[d].begin(); citr != d_shape[d].end(); ++citr)
        {
            center_of_mass[d] += *citr;
        }
    }

    for (int d = 0; d < NDIM; ++d) center_of_mass[d] /= total_lag_pts;

    // Shift COM to origin to apply rotation
    for (int d = 0; d < NDIM; ++d)
    {
        for (std::vector<double>::iterator itr = d_shape[d].begin(); itr != d_shape[d].end(); ++itr)
        {
            *itr -= center_of_mass[d];
        }
    }

    // Rotate shape about origin or center of mass
    const double angleFromHorizontal = d_initAngle_bodyAxis_x + incremented_angle_from_reference_axis[2];
    for (int i = 0; i < total_lag_pts; ++i)
    {
        const double x_rotated = d_shape[0][i] * cos(angleFromHorizontal) - d_shape[1][i] * sin(angleFromHorizontal);
        const double y_rotated = d_shape[0][i] * sin(angleFromHorizontal) + d_shape[1][i] * cos(angleFromHorizontal);
        d_shape[0][i] = x_rotated;
        d_shape[1][i] = y_rotated;
    }

    d_current_time = d_new_time;

    return;
}

void
IBEELKinematics::setEelSpecificVelocity(const double time,
                                        const std::vector<double>& incremented_angle_from_reference_axis,
                                        const std::vector<double>& center_of_mass,
                                        const std::vector<double>& tagged_pt_position)
{
    d_parser_time = time;
    const double angleFromHorizontal = d_initAngle_bodyAxis_x + incremented_angle_from_reference_axis[2];

    if (d_bodyIsManeuvering)
    {
        if (d_maneuverAxisIsChangingShape)
        {
            // Calculate radius of circular path for fish backbone
            double radius_circular_path;
            std::vector<double> bodyline_vector(NDIM), foodline_vector(NDIM);
            double mag_bodyline_vector = 0.0, mag_foodline_vector = 0.0;

            for (int dim = 0; dim < NDIM; ++dim)
            {
                bodyline_vector[dim] = tagged_pt_position[dim] - center_of_mass[dim];
                foodline_vector[dim] = d_food_location[dim] - tagged_pt_position[dim];
                mag_bodyline_vector += std::pow(bodyline_vector[dim], 2);
                mag_foodline_vector += std::pow(foodline_vector[dim], 2);
            }

            // Normalize vectors
            for (int dim = 0; dim < NDIM; ++dim)
            {
                bodyline_vector[dim] /= sqrt(mag_bodyline_vector);
                foodline_vector[dim] /= sqrt(mag_foodline_vector);
            }

            // Find angle between bodyline_axis and foodline_axis
            const double angle_bw_target_vision =
                sign(bodyline_vector[0] * foodline_vector[1] - bodyline_vector[1] * foodline_vector[0]) *
                std::acos(bodyline_vector[0] * foodline_vector[0] + bodyline_vector[1] * foodline_vector[1]);

            if (angle_bw_target_vision >= CUT_OFF_ANGLE)
            {
                radius_circular_path = CUT_OFF_RADIUS;
            }
            else if (angle_bw_target_vision <= -CUT_OFF_ANGLE)
            {
                radius_circular_path = CUT_OFF_RADIUS;
            }
            else if (IBTK::abs_equal_eps(MathUtilities<double>::Abs(angle_bw_target_vision), 0.0))
            {
                radius_circular_path = __INFINITY;
            }
            else if (angle_bw_target_vision >= -LOWER_CUT_OFF_ANGLE && angle_bw_target_vision <= LOWER_CUT_OFF_ANGLE)
            {
                radius_circular_path = std::abs(CUT_OFF_RADIUS * std::pow((CUT_OFF_ANGLE / LOWER_CUT_OFF_ANGLE), 1));
            }
            else
            {
                radius_circular_path = std::abs(CUT_OFF_RADIUS * std::pow((CUT_OFF_ANGLE / angle_bw_target_vision), 1));
            }
            
            // Set reference maneuver axis coordinates
            const int BodyNx = static_cast<int>(ceil(LENGTH_FISH / d_mesh_width[0]));
            if (radius_circular_path != __INFINITY)
            {
                const double angle_sector = LENGTH_FISH / radius_circular_path;
                const double dtheta = angle_sector / (BodyNx - 1);

                d_maneuverAxisReferenceCoordinates_vec.clear();
                std::vector<double> vec_axis_coord(2);
                for (int i = 1; i <= BodyNx; ++i)
                {
                    const double angleFromVertical = -angle_sector / 2 + (i - 1) * dtheta;
                    vec_axis_coord[0] = radius_circular_path * sin(angleFromVertical);
                    vec_axis_coord[1] = radius_circular_path * cos(angleFromVertical);
                    d_maneuverAxisReferenceCoordinates_vec.push_back(vec_axis_coord);
                }
            }
            else
            {
                d_maneuverAxisReferenceCoordinates_vec.clear();
                std::vector<double> vec_axis_coord(2);
                for (int i = 1; i <= BodyNx; ++i)
                {
                    vec_axis_coord[0] = (i - 1) * d_mesh_width[0];
                    vec_axis_coord[1] = 0.0;
                    d_maneuverAxisReferenceCoordinates_vec.push_back(vec_axis_coord);
                }
            }

            // Find COM of maneuver axis
            double maneuverAxis_x_cm = 0.0;
            double maneuverAxis_y_cm = 0.0;
            for (unsigned int i = 0; i < d_maneuverAxisReferenceCoordinates_vec.size(); ++i)
            {
                maneuverAxis_x_cm += d_maneuverAxisReferenceCoordinates_vec[i][0];
                maneuverAxis_y_cm += d_maneuverAxisReferenceCoordinates_vec[i][1];
            }
            maneuverAxis_x_cm /= d_maneuverAxisReferenceCoordinates_vec.size();
            maneuverAxis_y_cm /= d_maneuverAxisReferenceCoordinates_vec.size();

            // Shift reference so maneuver axis COM coincides with origin
            for (unsigned int i = 0; i < d_maneuverAxisReferenceCoordinates_vec.size(); ++i)
            {
                d_maneuverAxisReferenceCoordinates_vec[i][0] -= maneuverAxis_x_cm;
                d_maneuverAxisReferenceCoordinates_vec[i][1] -= maneuverAxis_y_cm;
            }

            // Find tangents on reference axis for shape update
            d_map_reference_tangent.clear();
            d_map_reference_sign.clear();
            for (int i = 0; i <= (BodyNx - 2); ++i)
            {
                std::vector<int> sign_vec(2);
                const double s = i * d_mesh_width[0];
                const double dX =
                    (d_maneuverAxisReferenceCoordinates_vec[i + 1][0] - d_maneuverAxisReferenceCoordinates_vec[i][0]);
                const double dY =
                    (d_maneuverAxisReferenceCoordinates_vec[i + 1][1] - d_maneuverAxisReferenceCoordinates_vec[i][1]);
                sign_vec[0] = sign(dX);
                sign_vec[1] = sign(dY);
                const double theta = std::atan(std::abs(dY / dX));
                d_map_reference_tangent.insert(std::make_pair(s, theta));
                d_map_reference_sign.insert(std::make_pair(s, sign_vec));
            }
            
            // Fill in last point
            d_map_reference_tangent.insert(
                std::make_pair((BodyNx - 1) * d_mesh_width[0], (d_map_reference_tangent.rbegin())->second));
            d_map_reference_sign.insert(
                std::make_pair((BodyNx - 1) * d_mesh_width[0], (d_map_reference_sign.rbegin())->second));
        }

        // Rotate reference axis and calculate tangents in rotated frame
        transformManeuverAxisAndCalculateTangents(angleFromHorizontal);
    }

    // Use numerically computed deformation velocity
    std::vector<double> vec_vel(NDIM);
    int lag_idx = 0;
    for (std::map<double, int>::const_iterator itr = d_ImmersedBodyData.begin(); itr != d_ImmersedBodyData.end(); itr++)
    {
        const double s_l = itr->first;  // Arc length coordinate (X_0)
        const int NumPtsInSection = itr->second;

        // Calculate deformation velocity at this point
        double deform_vel = calculateDeformationVelocity(s_l, time);
        
        // Set normal vector
        if (d_bodyIsManeuvering)
        {
            d_parser_normal[0] =
                -sin(d_map_transformed_tangent[s_l]) * d_map_transformed_sign[s_l][1];
            d_parser_normal[1] =
                cos(d_map_transformed_tangent[s_l]) * d_map_transformed_sign[s_l][0];
        }
        else
        {
            d_parser_normal[0] = -sin(angleFromHorizontal);
            d_parser_normal[1] = cos(angleFromHorizontal);
        }

        // Decompose deformation velocity into x and y directions
        vec_vel[0] = deform_vel * d_parser_normal[0];
        vec_vel[1] = deform_vel * d_parser_normal[1];

        // Set same velocity for all points in this section
        const int lowerlimit = lag_idx;
        const int upperlimit = lag_idx + NumPtsInSection;
        for (int d = 0; d < NDIM; ++d)
        {
            for (int i = lowerlimit; i < upperlimit; ++i) d_kinematics_vel[d][i] = vec_vel[d];
        }

        lag_idx = upperlimit;
    }

    return;
}

void
IBEELKinematics::setKinematicsVelocity(const double time,
                                       const std::vector<double>& incremented_angle_from_reference_axis,
                                       const std::vector<double>& center_of_mass,
                                       const std::vector<double>& tagged_pt_position)
{
    d_new_time = time;
    d_incremented_angle_from_reference_axis = incremented_angle_from_reference_axis;
    d_center_of_mass = center_of_mass;
    d_tagged_pt_position = tagged_pt_position;

    setEelSpecificVelocity(d_new_time, d_incremented_angle_from_reference_axis, d_center_of_mass, d_tagged_pt_position);

    return;

} // setNewKinematicsVelocity

const std::vector<std::vector<double> >&
IBEELKinematics::getKinematicsVelocity(const int /*level*/) const
{
    return d_kinematics_vel;

} // getKinematicsVelocity

const std::vector<std::vector<double> >&
IBEELKinematics::getShape(const int /*level*/) const
{
    return d_shape;
} // getShape

void
IBEELKinematics::calculateCoefficients(double theta_lmax_prev, double theta_lmax_next, 
                                     double lambda_prev, double lambda_next)
{
    // Calculate half wavelength
    double half_lambda = lambda_next / 2.0;
    
    // Set boundary condition equation system
    // Equation 1: c0 = theta_lmax_prev
    d_c0 = theta_lmax_prev;
    
    // Equation 3: c1 = 0
    d_c1 = 0.0;
    
    // Equation 5: 2*c2 = -theta_lmax_prev*(2*pi/lambda_prev)^2
    d_c2 = -theta_lmax_prev * pow(2.0 * M_PI / lambda_prev, 2) / 2.0;
    
    // Solve remaining coefficients using linear system
    // Equation 2: c0 + c1*(lambda_next/2) + c2*(lambda_next/2)^2 + c3*(lambda_next/2)^3 + c4*(lambda_next/2)^4 + c5*(lambda_next/2)^5 = theta_lmax_next
    // Equation 4: c1 + 2*c2*(lambda_next/2) + 3*c3*(lambda_next/2)^2 + 4*c4*(lambda_next/2)^3 + 5*c5*(lambda_next/2)^4 = 0
    // Equation 6: 2*c2 + 6*c3*(lambda_next/2) + 12*c4*(lambda_next/2)^2 + 20*c5*(lambda_next/2)^3 = -theta_lmax_next*(2*pi/lambda_next)^2
    
    // 3x3 linear system for c3, c4, c5
    
    // Equation 2 simplified: c3*(lambda_next/2)^3 + c4*(lambda_next/2)^4 + c5*(lambda_next/2)^5 = theta_lmax_next - c0 - c2*(lambda_next/2)^2
    double b1 = theta_lmax_next - d_c0 - d_c2 * pow(half_lambda, 2);
    
    // Equation 4 simplified: 3*c3*(lambda_next/2)^2 + 4*c4*(lambda_next/2)^3 + 5*c5*(lambda_next/2)^4 = -2*c2*(lambda_next/2)
    double b2 = -2.0 * d_c2 * half_lambda;
    
    // Equation 6 simplified: 6*c3*(lambda_next/2) + 12*c4*(lambda_next/2)^2 + 20*c5*(lambda_next/2)^3 = -theta_lmax_next*(2*pi/lambda_next)^2 - 2*c2
    double b3 = -theta_lmax_next * pow(2.0 * M_PI / lambda_next, 2) - 2.0 * d_c2;
    
    // Coefficient matrix
    double A[3][3] = {
        {pow(half_lambda, 3), pow(half_lambda, 4), pow(half_lambda, 5)},
        {3.0 * pow(half_lambda, 2), 4.0 * pow(half_lambda, 3), 5.0 * pow(half_lambda, 4)},
        {6.0 * half_lambda, 12.0 * pow(half_lambda, 2), 20.0 * pow(half_lambda, 3)}
    };
    
    // Right-hand side vector
    double b[3] = {b1, b2, b3};
    
    // Gaussian elimination
    // Forward elimination
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 3; j++) {
            double factor = A[j][i] / A[i][i];
            b[j] -= factor * b[i];
            for (int k = i; k < 3; k++) {
                A[j][k] -= factor * A[i][k];
            }
        }
    }
    
    // Back substitution
    d_c5 = b[2] / A[2][2];
    d_c4 = (b[1] - A[1][2] * d_c5) / A[1][1];
    d_c3 = (b[0] - A[0][1] * d_c4 - A[0][2] * d_c5) / A[0][0];
    
    // Print calculated coefficients for debugging
    plog << "Calculated coefficients: c0=" << d_c0 << ", c1=" << d_c1 << ", c2=" << d_c2 
         << ", c3=" << d_c3 << ", c4=" << d_c4 << ", c5=" << d_c5 << std::endl;
}

void
IBEELKinematics::updateCoefficients(double theta_lmax_prev, double theta_lmax_next, 
                                  double lambda_prev, double lambda_next, 
                                  double current_time, double new_T_n)
{
    // 添加更详细的调试信息
    if (IBTK_MPI::getRank() == 0) {
        pout << "\n=== 系数更新前（时间=" << d_parser_time << ", t_offset=" << d_time_offset << "）===\n";
        
        // 打印当前几个样本点的状态
        double sample_points[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
        
        pout << "更新前形状采样（s_l, y_shape_base=calculateHl(s_l, d_parser_time)):\n";
        for (int i = 0; i < 5; i++) {
            double s_l = sample_points[i] * LENGTH_FISH;
            double h_l = calculateHl(s_l, d_parser_time);
            pout << "s_l=" << s_l << ", y_shape_base=" << h_l << "\n";
        }
        
        pout << "更新前变形速度采样（s_l, vel=calculateDeformationVelocity(s_l, d_parser_time)):\n";
        for (int i = 0; i < 5; i++) {
            double s_l = sample_points[i] * LENGTH_FISH;
            double vel = calculateDeformationVelocity(s_l, d_parser_time);
            pout << "s_l=" << s_l << ", vel=" << vel << "\n";
        }
    }
    
    // 首次调用时初始化缓存
    bool first_time = !d_cache_initialized;
    if (first_time) {
        if (IBTK_MPI::getRank() == 0) {
            pout << "调试: 首次调用updateCoefficients，即将初始化缓存\n";
        }
        initializeCache(theta_lmax_prev, theta_lmax_next, lambda_prev, lambda_next);
        
        if (IBTK_MPI::getRank() == 0) {
            pout << "========= 当前缓存的所有系数（时间=" << d_parser_time << "）=========\n";
            
            // 打印缓存信息...
            for (int i = 0; i < 3; i++) {
                pout << "缓存[" << i << "]: theta_lmax_prev=" << d_coefficient_cache[i].theta_lmax_prev
                     << ", theta_lmax_next=" << d_coefficient_cache[i].theta_lmax_next << "\n";
                pout << "    c0=" << d_coefficient_cache[i].c0
                     << ", c1=" << d_coefficient_cache[i].c1
                     << ", c2=" << d_coefficient_cache[i].c2
                     << ", c3=" << d_coefficient_cache[i].c3
                     << ", c4=" << d_coefficient_cache[i].c4
                     << ", c5=" << d_coefficient_cache[i].c5 << "\n";
            }
            
            pout << "========================================================\n";
        }
        
        return;
    }
    
    // 使用从main.cpp传入的current_time直接赋值给d_time_offset
    d_time_offset = current_time;
    
    // 在第826行附近添加T_n更新
    d_wave_function_T_n = new_T_n;
    
    if (IBTK_MPI::getRank() == 0) {
        pout << "时间偏移量更新: d_time_offset = " << d_time_offset << " (使用传入的current_time)\n";
        pout << "T_n更新: d_wave_function_T_n = " << d_wave_function_T_n << "\n";
    }
    
    // 滚动更新缓存 - 移动现有的缓存内容
    d_coefficient_cache[0] = d_coefficient_cache[1];
    d_coefficient_cache[1] = d_coefficient_cache[2];
    
    // 计算新的系数并存入第三个缓存位置
    calculateCoefficients(theta_lmax_prev, theta_lmax_next, lambda_prev, lambda_next);
    d_coefficient_cache[2].theta_lmax_prev = theta_lmax_prev;
    d_coefficient_cache[2].theta_lmax_next = theta_lmax_next;
    d_coefficient_cache[2].lambda_prev = lambda_prev;
    d_coefficient_cache[2].lambda_next = lambda_next;
    d_coefficient_cache[2].c0 = d_c0;
    d_coefficient_cache[2].c1 = d_c1;
    d_coefficient_cache[2].c2 = d_c2;
    d_coefficient_cache[2].c3 = d_c3;
    d_coefficient_cache[2].c4 = d_c4;
    d_coefficient_cache[2].c5 = d_c5;
    
    // 在主进程上打印所有缓存的系数
    if (IBTK_MPI::getRank() == 0) {
        pout << "========= 当前缓存的所有系数（时间=" << d_parser_time << ", 新t_offset=" << d_time_offset << "）=========\n";
        
        for (int i = 0; i < 3; i++) {
            const char* cache_names[] = {"最旧", "中间", "最新"};
            pout << "缓存[" << i << "]（" << cache_names[i] << "）: theta_lmax_prev=" << d_coefficient_cache[i].theta_lmax_prev
                 << ", theta_lmax_next=" << d_coefficient_cache[i].theta_lmax_next << "\n";
            pout << "    c0=" << d_coefficient_cache[i].c0
                 << ", c1=" << d_coefficient_cache[i].c1
                 << ", c2=" << d_coefficient_cache[i].c2
                 << ", c3=" << d_coefficient_cache[i].c3
                 << ", c4=" << d_coefficient_cache[i].c4
                 << ", c5=" << d_coefficient_cache[i].c5 << "\n";
        }
        
        pout << "========================================================\n";
        
        // 打印更新后的样本点状态
        double sample_points[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
        
        pout << "更新后形状采样（s_l, y_shape_base=calculateHl(s_l, d_parser_time)):\n";
        for (int i = 0; i < 5; i++) {
            double s_l = sample_points[i] * LENGTH_FISH;
            double h_l = calculateHl(s_l, d_parser_time);
            pout << "s_l=" << s_l << ", y_shape_base=" << h_l << "\n";
        }
        
        pout << "更新后变形速度采样（s_l, vel=calculateDeformationVelocity(s_l, d_parser_time)):\n";
        for (int i = 0; i < 5; i++) {
            double s_l = sample_points[i] * LENGTH_FISH;
            double vel = calculateDeformationVelocity(s_l, d_parser_time);
            pout << "s_l=" << s_l << ", vel=" << vel << "\n";
        }
        
        pout << "=== 系数更新完成 ===\n\n";
    }
}

double
IBEELKinematics::calculateThetaL(double s_l, double t)
{
    // 使用新的t_rel计算方式：t_rel = t - t_offset
    double t_rel = t - d_time_offset;
    

    
    // 计算zeta值
    double zeta = (d_wave_function_lambda / d_wave_function_T_n) * t_rel - (s_l / LENGTH_FISH);
    
    // 使用波形函数计算基础theta_l值
    double base_theta_l = waveformFunction(zeta);
    
    // 添加(s_l/L)^2的系数，表示沿鱼体长度的振幅变化
    double amplitude_factor = pow(s_l / LENGTH_FISH, 2);
    double theta_l = base_theta_l * amplitude_factor;
    
    // 检查计算结果是否有效
    if (std::isnan(theta_l) || std::isinf(theta_l)) {
        if (IBTK_MPI::getRank() == 0) {
            pout << "警告: calculateThetaL 计算结果无效! s_l=" << s_l << ", t=" << t 
                 << ", t_rel=" << t_rel << ", zeta=" << zeta << ", theta_l=" << theta_l << "\n";
            pout << "  base_theta_l=" << base_theta_l 
                 << ", amplitude_factor=" << amplitude_factor << "\n";
        }
        return 0.0;
    }
    
    return theta_l;
}

double 
IBEELKinematics::integrandSinTheta(double s, double t)
{
    // 计算被积函数sin(θₗ(s,t))
    double theta_l = calculateThetaL(s, t);
    return sin(theta_l);
}

double
IBEELKinematics::calculateHl(double s_l, double t)
{
    if (d_use_exact_integration) {
        // 使用复合梯形积分法进行数值积分
        double result = 0.0;
        double step = d_mesh_width[0]; // Use grid spacing for consistency
        int n = static_cast<int>(round(s_l / step)); // Use round for floating point safety
        
        // 如果积分上限太小，直接返回0
        if (n == 0) return 0.0;
        
        // 复合梯形公式: ∫a^b f(x)dx ≈ (b-a)/2n * [f(a) + 2*f(a+h) + 2*f(a+2h) + ... + 2*f(a+(n-1)h) + f(b)]
        result += integrandSinTheta(0.0, t);                 // f(a)
        result += integrandSinTheta(s_l, t);                 // f(b)
        
        for (int i = 1; i < n; ++i) {
            double s = i * step;
            result += 2.0 * integrandSinTheta(s, t);         // 2*f(a+i*h)
        }
        
        result *= step / 2.0;
        
        // 检查计算结果是否有效
        if (std::isnan(result) || std::isinf(result)) {
            if (IBTK_MPI::getRank() == 0) {
                pout << "警告: calculateHl 计算结果无效! s_l=" << s_l << ", t=" << t 
                     << ", result=" << result << "\n";
                
                // 打印积分步骤的值以帮助诊断
                pout << "  integrandSinTheta(0.0, t)=" << integrandSinTheta(0.0, t) << "\n";
                pout << "  integrandSinTheta(s_l, t)=" << integrandSinTheta(s_l, t) << "\n";
                for (int i = 1; i < std::min(n, 5); ++i) {
                    double s = i * step;
                    pout << "  integrandSinTheta(" << s << ", t)=" << integrandSinTheta(s, t) << "\n";
                }
            }
            return 0.0;  // 返回一个安全的默认值
        }
        
        return result;
    } else {
        // 使用多项式近似或其他快速方法
        // 这里可以根据需要实现更高效的近似方法
        // 简单起见，我们暂时使用简化的多项式近似
        double theta_l = calculateThetaL(s_l / 2.0, t);      // 在中点处的值
        return s_l * sin(theta_l);                          // 粗略近似
    }
}

double
IBEELKinematics::calculateDeformationVelocity(double s_l, double t)
{
    // 使用中心差分法计算时间导数：∂h_l/∂t ≈ (h_l(s_l,t+dt) - h_l(s_l,t-dt))/(2*dt)
    double dt = d_time_step_size;
    double h_l_forward = calculateHl(s_l, t + dt);
    double h_l_backward = calculateHl(s_l, t - dt);
    
    double result = (h_l_forward - h_l_backward) / (2.0 * dt);
    
    // 检查计算结果是否有效
    if (std::isnan(result) || std::isinf(result)) {
        if (IBTK_MPI::getRank() == 0) {
            pout << "警告: calculateDeformationVelocity 计算结果无效! s_l=" << s_l << ", t=" << t 
                 << ", result=" << result << "\n";
            pout << "  h_l_forward=" << h_l_forward << ", h_l_backward=" << h_l_backward 
                 << ", dt=" << dt << "\n";
        }
        return 0.0;  // 返回一个安全的默认值
    }
    
    return result;
}

// 初始化缓存器方法的实现
void 
IBEELKinematics::initializeCache(double theta_lmax_prev, double theta_lmax_next, 
                               double lambda_prev, double lambda_next)
{
    if (d_cache_initialized) return;
    
    d_coefficient_cache.resize(3);
    
    // 初始化第一组系数 (最老的一组)
    calculateCoefficients(theta_lmax_prev, theta_lmax_next, lambda_prev, lambda_next);
    d_coefficient_cache[0].theta_lmax_prev = theta_lmax_prev;
    d_coefficient_cache[0].theta_lmax_next = theta_lmax_next;
    d_coefficient_cache[0].lambda_prev = lambda_prev;
    d_coefficient_cache[0].lambda_next = lambda_next;
    d_coefficient_cache[0].c0 = d_c0;
    d_coefficient_cache[0].c1 = d_c1;
    d_coefficient_cache[0].c2 = d_c2;
    d_coefficient_cache[0].c3 = d_c3;
    d_coefficient_cache[0].c4 = d_c4;
    d_coefficient_cache[0].c5 = d_c5;
    
    // 初始化第二组系数 (倒数第二组)，交换theta_lmax_prev和theta_lmax_next
    calculateCoefficients(theta_lmax_next, theta_lmax_prev, lambda_prev, lambda_next);
    d_coefficient_cache[1].theta_lmax_prev = theta_lmax_next;
    d_coefficient_cache[1].theta_lmax_next = theta_lmax_prev;
    d_coefficient_cache[1].lambda_prev = lambda_prev;
    d_coefficient_cache[1].lambda_next = lambda_next;
    d_coefficient_cache[1].c0 = d_c0;
    d_coefficient_cache[1].c1 = d_c1;
    d_coefficient_cache[1].c2 = d_c2;
    d_coefficient_cache[1].c3 = d_c3;
    d_coefficient_cache[1].c4 = d_c4;
    d_coefficient_cache[1].c5 = d_c5;
    
    // 初始化第三组系数 (最新的一组)
    calculateCoefficients(theta_lmax_prev, theta_lmax_next, lambda_prev, lambda_next);
    d_coefficient_cache[2].theta_lmax_prev = theta_lmax_prev;
    d_coefficient_cache[2].theta_lmax_next = theta_lmax_next;
    d_coefficient_cache[2].lambda_prev = lambda_prev;
    d_coefficient_cache[2].lambda_next = lambda_next;
    d_coefficient_cache[2].c0 = d_c0;
    d_coefficient_cache[2].c1 = d_c1;
    d_coefficient_cache[2].c2 = d_c2;
    d_coefficient_cache[2].c3 = d_c3;
    d_coefficient_cache[2].c4 = d_c4;
    d_coefficient_cache[2].c5 = d_c5;
    
    d_cache_initialized = true;
    
    if (IBTK_MPI::getRank() == 0) {
        pout << "缓存器初始化完成：\n"
             << "缓存[0]: theta_lmax_prev=" << d_coefficient_cache[0].theta_lmax_prev 
             << ", theta_lmax_next=" << d_coefficient_cache[0].theta_lmax_next << "\n"
             << "缓存[1]: theta_lmax_prev=" << d_coefficient_cache[1].theta_lmax_prev 
             << ", theta_lmax_next=" << d_coefficient_cache[1].theta_lmax_next << "\n"
             << "缓存[2]: theta_lmax_prev=" << d_coefficient_cache[2].theta_lmax_prev 
             << ", theta_lmax_next=" << d_coefficient_cache[2].theta_lmax_next << "\n";
    }
}

// 获取缓存中的系数方法实现
void 
IBEELKinematics::getCachedCoefficients(int cache_index)
{
    // 如果缓存未初始化，立即初始化它
    if (!d_cache_initialized) {
        if (IBTK_MPI::getRank() == 0) {
            pout << "警告: 在getCachedCoefficients中检测到缓存未初始化，使用默认值初始化\n";
        }
        // 使用默认值初始化缓存
        //initializeCache(-1.0, 1.0, d_wave_function_lambda, d_wave_function_lambda);
        // 直接人工反向！！！！！！  正常时-1 ，1， 对应main中-1， 1
        initializeCache(1.0, -1.0, d_wave_function_lambda, d_wave_function_lambda);
    }
    
    
    // 检查索引是否有效
    if (cache_index < 0 || cache_index >= d_coefficient_cache.size()) {
        if (IBTK_MPI::getRank() == 0) {
            pout << "警告: 缓存索引超出范围，使用默认索引2\n";
        }
        cache_index = 2;  // 使用默认索引，避免崩溃
    }
    
    // 获取缓存的系数
    d_c0 = d_coefficient_cache[cache_index].c0;
    d_c1 = d_coefficient_cache[cache_index].c1;
    d_c2 = d_coefficient_cache[cache_index].c2;
    d_c3 = d_coefficient_cache[cache_index].c3;
    d_c4 = d_coefficient_cache[cache_index].c4;
    d_c5 = d_coefficient_cache[cache_index].c5;
}

double 
IBEELKinematics::waveformFunction(double zeta)
{
    double phase_adjust = 0.0;
    int cache_index = 2;  // 默认使用最新的系数
    
    if (zeta > 0) {
        // zeta > 0: 使用最新的系数（索引2）
        cache_index = 2;
        phase_adjust = 0.0;
    } else if (zeta >= -d_wave_function_lambda/2.0) {
        // -lambda_val/2 <= zeta <= 0: 使用中间的系数（索引1）
        cache_index = 1;
        phase_adjust = d_wave_function_lambda/2.0;
    } else {
        // zeta < -lambda_val/2: 使用最旧的系数（索引0）
        cache_index = 0;
        phase_adjust = d_wave_function_lambda;
    }
    

    
    // 从缓存中获取对应的系数
    getCachedCoefficients(cache_index);
    
    // 调整zeta值以匹配相应半周期的相位
    double adjusted_zeta = zeta + phase_adjust;
    
    // 计算并返回波形函数值
    double result = d_c0 + d_c1 * adjusted_zeta + d_c2 * pow(adjusted_zeta, 2) + 
           d_c3 * pow(adjusted_zeta, 3) + d_c4 * pow(adjusted_zeta, 4) + 
           d_c5 * pow(adjusted_zeta, 5);
    
    // 检查计算结果是否有效
    if (std::isnan(result) || std::isinf(result)) {
        if (IBTK_MPI::getRank() == 0) {
            pout << "警告: waveformFunction 计算结果无效! zeta=" << zeta 
                 << ", adjusted_zeta=" << adjusted_zeta 
                 << ", result=" << result << "\n";
            pout << "  cache_index=" << cache_index 
                 << ", c0=" << d_c0 << ", c1=" << d_c1 << ", c2=" << d_c2 
                 << ", c3=" << d_c3 << ", c4=" << d_c4 << ", c5=" << d_c5 << "\n";
        }
        return 0.0;  // 返回一个安全的默认值
    }
    
    return result;
}

// New function to calculate the integrand cos(theta_l(s,t))
double
IBEELKinematics::integrandCosTheta(double s, double t)
{
    // Calculate the integrand cos(θₗ(s,t))
    double theta_l = calculateThetaL(s, t);
    return cos(theta_l);
}

// New function to calculate x_l(s_l, t) = ∫₀ˢˡ cos(θₗ(s,t)) ds
double
IBEELKinematics::calculateXl(double s_l, double t)
{
    if (d_use_exact_integration) {
        // Using composite trapezoidal rule for numerical integration
        double result = 0.0;
        double step = d_mesh_width[0]; // Use grid spacing for consistency
        int n = static_cast<int>(round(s_l / step)); // Use round for floating point safety
        
        // If the integration upper limit is too small (s_l < step), return 0.0
        // This is consistent with calculateHl and correct for s_l = 0.
        if (n == 0) return 0.0; 
        
        result += integrandCosTheta(0.0, t);                 // f(a)
        result += integrandCosTheta(s_l, t);                 // f(b)
        
        for (int i = 1; i < n; ++i) {
            double s_prime = i * step;
            result += 2.0 * integrandCosTheta(s_prime, t);   // 2*f(a+i*h)
        }
        
        result *= step / 2.0;
        
        // Check if the calculation result is valid
        if (std::isnan(result) || std::isinf(result)) {
            if (IBTK_MPI::getRank() == 0) {
                pout << "Warning: calculateXl calculation result is invalid! s_l=" << s_l << ", t=" << t 
                     << ", result=" << result << "\n";
            }
            // Fallback value: return s_l, approximating a straight line where x is the arc length.
            // This is generally safer than returning 0.0 if s_l > 0.
            return s_l;  
        }
        
        return result;
    } else {
        // Use polynomial approximation or other fast methods
        // For simplicity, using a simplified approximation similar to calculateHl's else branch
        double theta_l_mid = calculateThetaL(s_l / 2.0, t);      // Value at the midpoint
        return s_l * cos(theta_l_mid);                          // Rough approximation
    }
}

} // namespace IBAMR
