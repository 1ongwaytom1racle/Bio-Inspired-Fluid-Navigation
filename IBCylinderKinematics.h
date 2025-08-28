#ifndef included_IBCylinderKinematics
#define included_IBCylinderKinematics

/////////////////////////////// INCLUDES /////////////////////////////////////

// IBAMR INCLUDES
#include "ibamr/ConstraintIBKinematics.h"

// C++ INCLUDES
#include <string>
#include <vector>

/////////////////////////////////////// FORWARD DECLARATION ////////////////////////////////

namespace mu
{
class Parser;
} // namespace mu

namespace IBAMR
{
/*!
 * \brief Class IBCylinderKinematics provides definition for cylinder kinematics.
 */
class IBCylinderKinematics : public ConstraintIBKinematics
{
public:
    /*!
     * \brief Constructor.
     */
    IBCylinderKinematics(const std::string& object_name,
                         SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                         IBTK::LDataManager* l_data_manager,
                         SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM> > patch_hierarchy,
                         bool register_for_restart = true);

    /*!
     * \brief Destructor.
     */
    ~IBCylinderKinematics();

    /*!
     * \brief Set kinematics velocity at new time for cylinder.
     */
    void setKinematicsVelocity(const double time,
                              const std::vector<double>& incremented_angle_from_reference_axis,
                              const std::vector<double>& center_of_mass,
                              const std::vector<double>& tagged_pt_position) override;

    /*!
     * \brief Get the kinematics velocity at new time for cylinder on the specified level.
     */
    const std::vector<std::vector<double> >& getKinematicsVelocity(const int level) const override;

    /*!
     * \brief Set the shape of cylinder at new time on all levels.
     */
    void setShape(const double time, const std::vector<double>& incremented_angle_from_reference_axis) override;

    /*!
     * \brief Get the shape of cylinder at new time on the specified level.
     */
    const std::vector<std::vector<double> >& getShape(const int level) const override;

    /*!
     * \brief Override the base Serializable method.
     */
    void putToDatabase(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> db) override;

private:
    /*!
     * \brief Get necessary data from restart manager for restarted runs.
     */
    void getFromRestart();

    /*!
     * \brief Set cylinder specific velocity.
     */
    void setCylinderSpecificVelocity(const double time,
                                    const std::vector<double>& incremented_angle_from_reference_axis,
                                    const std::vector<double>& center_of_mass,
                                    const std::vector<double>& tagged_pt_position);

    /*!
     * The mu::Parser objects which evaluate the data-setting functions.
     */
    std::vector<mu::Parser*> d_kinematicsvel_parsers;
    std::vector<mu::Parser*> d_all_parsers;

    /*!
     * Input kinematics velocity functions.
     */
    std::vector<std::string> d_kinematicsvel_function_strings;

    /*!
     * Parser variables.
     */
    double d_parser_time;
    IBTK::Point d_parser_posn;

    /*!
     * Current time (t) and new time (t+dt).
     */
    double d_current_time;
    double d_new_time;

    /*!
     * New kinematics velocity. New shape of the body.
     */
    std::vector<std::vector<std::vector<double> > > d_kinematics_vel;
    std::vector<std::vector<std::vector<double> > > d_shape;

    /*!
     * Save COM, tagged point position and incremented angle from reference axis for restarted runs.
     */
    std::vector<double> d_center_of_mass;
    std::vector<double> d_incremented_angle_from_reference_axis;
    std::vector<double> d_tagged_pt_position;
};

} // namespace IBAMR

#endif //#ifndef included_IBCylinderKinematics 