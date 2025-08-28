// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2023 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

#ifndef included_IBEELKinematics
#define included_IBEELKinematics

/////////////////////////////////////// INCLUDES ////////////////////////////////
#include "ibamr/ConstraintIBKinematics.h"

#include "PatchHierarchy.h"
#include "tbox/Array.h"
#include "tbox/Database.h"
#include "tbox/Pointer.h"

#include <iostream>
#include <map>
#include <vector>

namespace mu
{
class Parser;
} // namespace mu

///////////////////////////////////////////////////////////////// CLASS DEFORMATIONAL KINEMATICS //////////////////

namespace IBAMR
{
/*!
 * \brief IBEELKinematics is a concrete class which calculates the deformation velocity and updated shape
 * for 2D eel. It also provides routines for maneuvering and food tracking cases. Example taken from:
 *
 *  Bhalla et al. A unified mathematical framework and an adaptive numerical method for
 *  fluid-structure interaction with rigid, deforming, and elastic bodies. J Comput Phys, 250:446-476 (2013).
 */

class IBEELKinematics : public ConstraintIBKinematics

{
public:
    /*!
     * \brief ctor. This is the only ctor for this object.
     */
    IBEELKinematics(const std::string& object_name,
                    SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                    IBTK::LDataManager* l_data_manager,
                    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM> > patch_hierarchy,
                    bool register_for_restart = true);

    /*!
     * \brief Destructor.
     */
    virtual ~IBEELKinematics();

    /*!
     * \brief Set kinematics velocity for eel.
     * \see IBAMR::ConstraintIBKinematics::setKinematicsVelocity
     */
    virtual void setKinematicsVelocity(const double time,
                                       const std::vector<double>& incremented_angle_from_reference_axis,
                                       const std::vector<double>& center_of_mass,
                                       const std::vector<double>& tagged_pt_position);

    /*!
     * \brief Get the kinematics velocity on the specified level.
     * \see IBAMR::ConstraintIBKinematics::getKinematicsVelocity
     */
    virtual const std::vector<std::vector<double> >& getKinematicsVelocity(const int level) const;

    /*!
     * \brief Set the shape of eel at the required time.
     * \see IBAMR::ConstraintIBKinematics::setShape
     */
    virtual void setShape(const double time, const std::vector<double>& incremented_angle_from_reference_axis);

    /*!
     * \brief Get the shape of eel at the required level.
     * \see IBAMR::ConstraintIBKinematics::getShape
     */
    virtual const std::vector<std::vector<double> >& getShape(const int level) const;

    /*!
     * \brief Override the ConstraintIBkinematics base class method.
     */
    virtual void putToDatabase(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> db);

    // 暴露给外部的计算系数方法 - 添加current_time参数
    void updateCoefficients(double theta_lmax_prev, double theta_lmax_next, 
                           double lambda_prev, double lambda_next, 
                           double current_time, double new_T_n);


private:
    /*!
     * \brief The default constructor is not implemented and should not be used.
     */
    IBEELKinematics();

    /*!
     * \brief The copy constructor is not implemented and should not be used.
     */
    IBEELKinematics(const IBEELKinematics& from);

    /*!
     * \brief The assignment operator is not implemented and should not be used.
     */
    IBEELKinematics& operator=(const IBEELKinematics& that);

    /*!
     * \brief Set data from restart.
     */
    void getFromRestart();

    /*!
     * \brief set eel body shape related data.
     */
    void setImmersedBodyLayout(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM> > patch_hierarchy);

    /*!
     * \brief Set deformation kinematics velocity of the eel.
     */
    void setEelSpecificVelocity(const double time,
                                const std::vector<double>& incremented_angle_from_reference_axis,
                                const std::vector<double>& center_of_mass,
                                const std::vector<double>& tagged_pt_position);

    /*!
     * \brief Rotate the maneuver axis and caluclate tangents in this orientation.
     */
    void transformManeuverAxisAndCalculateTangents(const double angleFromHorizontal);

    /*!
     * Current time (t) and new time (t+dt).
     */
    double d_current_time, d_new_time;

    /*!
     * Deformational velocity and shape vectors.
     */
    std::vector<std::vector<double> > d_kinematics_vel;
    std::vector<std::vector<double> > d_shape;

    /*!
     * Save COM, tagged point position and incremented angle from reference axis for restarted runs.
     */
    std::vector<double> d_center_of_mass, d_incremented_angle_from_reference_axis, d_tagged_pt_position;

    /*!
     * Eulerian Mesh width parameters.
     */
    std::vector<double> d_mesh_width;

    /*!
     * The following map is used to store eel body shape specific data.
     * The arc length 's' varies from 0 - 1. In the std::map  the arc length 's' is used as a key.
     * d_ImmersedBodyData is used to store the no of material points which represents a cross section. The
     * width of cross section of eel varies with arc length.
     */
    std::map<double, int> d_ImmersedBodyData;

    /*!
     * Initial orientation of the body axis
     */
    double d_initAngle_bodyAxis_x;

    /*!
     * Boolean value indicating if eel is maneuvering or not.
     * If eel is maneuvering then the traveling wave will be along a curved axis, otherwise it will be on a straight
     * line
     */
    bool d_bodyIsManeuvering;

    /*!
     * Boolean to indicate if shape of maneuver axis is changing. The maneuver axis will change shape in food tracking
     * cases.
     */
    bool d_maneuverAxisIsChangingShape;

    /*!
     * Vector of coordinates defining the axis of maneuvering. The reference axis will rotate with body omega.
     * The coordinates of the rotated maneuver axis is stored separately.
     */
    std::vector<std::vector<double> > d_maneuverAxisReferenceCoordinates_vec, d_maneuverAxisTransformedCoordinates_vec;

    /*!
     * map of tangents along the body/maneuver axis in rotated frame. The key used is arc length 's' and it stores only
     * the abs(theta).
     * Sign of tangent is stored separately. This is done to avoid a lot of if conditions needed to determine the
     * quadrant of the
     * angle.
     */
    std::map<double, double> d_map_transformed_tangent;

    /*!
     * map of tangents along the body/maneuver axis in reference/unrotated frame.The key used is arc length 's' and it
     * stores only the abs(theta).
     * Sign of tangent is stored separately. This is done to avoid a lot of if conditions needed to determine the
     * quadrant of the
     * angle.
     */
    std::map<double, double> d_map_reference_tangent;

    /*!
     * Sign of tangent vector in rotated frame. The key used is arc length 's'. 'mapped_value' is a vector which has
     * sign of t_x and t_y
     * respectively.
     */
    std::map<double, std::vector<int> > d_map_transformed_sign;

    /*!
     * Sign of tangent vector in reference/unrotated frame. The key used is arc length 's'. 'mapped_value' is a vector
     * which has sign of t_x and t_y
     * respectively.
     */
    std::map<double, std::vector<int> > d_map_reference_sign;

    /*!
     * mu::Parser object which evaluates the maneuvering axis equation.
     */
    mu::Parser* d_maneuvering_axis_parser;

    /*!
     * mu::Parser object which evaluates the shape of the body.
     */
    mu::Parser* d_body_shape_parser;

    /*!
     * The mu::Parser objects which evaluate the data-setting functions.
     */
    std::vector<mu::Parser*> d_deformationvel_parsers;
    std::vector<mu::Parser*> d_all_parsers;

    /*!
     * Time and position variables.
     */
    mutable double d_parser_time;
    mutable IBTK::Point d_parser_posn;
    mutable IBTK::Point d_parser_normal;

    /*!
     * Array containing initial coordinates of the food location.
     */
    SAMRAI::tbox::Array<double> d_food_location;

    // 新增：六个常系数用于复杂的身体形状方程
    double d_c0, d_c1, d_c2, d_c3, d_c4, d_c5;
    
    // 计算系数的函数
    void calculateCoefficients(double theta_lmax_prev, double theta_lmax_next, 
                              double lambda_prev, double lambda_next);

    // 计算h_l(s_l,t) = ∫₀^(s_l) sin(θₗ(s,t))ds
    double calculateHl(double s_l, double t);
    
    // 计算θₗ(s,t)的值
    double calculateThetaL(double s_l, double t);
    
    // 被积函数 sin(θₗ(s,t))
    double integrandSinTheta(double s, double t);
    
    // 数值积分所需的步长
    double d_integration_step_size;
    
    // 是否使用精确数值积分还是近似多项式
    bool d_use_exact_integration;

    // 计算变形速度 ∂h_l(s_l,t)/∂t
    double calculateDeformationVelocity(double s_l, double t);
    
    // 时间步长，用于数值求导
    double d_time_step_size;

    // 波形函数参数
    double d_wave_function_lambda;  // 波长
    double d_wave_function_T_n;     // 时间归一化因子

    // 系数缓存结构体
    struct CoefficientCacheEntry {
        double theta_lmax_prev;
        double theta_lmax_next;
        double lambda_prev;
        double lambda_next;
        double c0, c1, c2, c3, c4, c5;
    };
    
    // 系数缓存数组
    std::vector<CoefficientCacheEntry> d_coefficient_cache;
    
    // 缓存是否已初始化的标志
    bool d_cache_initialized;
    
    // 初始化缓存的方法
    void initializeCache(double theta_lmax_prev = 0.0, double theta_lmax_next = 0.0, 
                        double lambda_prev = 1.0, double lambda_next = 1.0);
    
    // 从缓存获取系数的方法
    void getCachedCoefficients(int cache_index);
    
    // 波形函数，根据zeta值计算波形
    double waveformFunction(double zeta);

    // 时间偏移量，用于控制t_rel的计算
    double d_time_offset;

    // New function to calculate the integrand cos(theta_l(s,t))
    double integrandCosTheta(double s, double t);

    // New function to calculate x_l(s_l, t) = ∫₀ˢˡ cos(θₗ(s,t)) ds
    double calculateXl(double s_l, double t);

}; // IBEELKinematics

} // namespace IBAMR
#endif // #ifndef included_IBEELKinematics
