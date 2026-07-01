#ifndef SLAM3D_ROCK_ODOMETRY_HPP
#define SLAM3D_ROCK_ODOMETRY_HPP

#include <slam3d/core/PoseSensor.hpp>
#include <slam3d/core/Solver.hpp>
#include <transformer/Transformer.hpp>

#include <vector>

namespace slam3d
{
	class RockOdometry : public PoseSensor
	{
	public:
		/**
		 * @brief Pose sensor that uses the Rock transformer
		 * @param name
		 * @param graph
		 * @param s
		 * @param logger
		 * @param tf
		 * @param interpolate set to true when align_port is used
		 */
		RockOdometry(const std::string& name, Graph* graph, Solver* s, Logger* logger,
			transformer::Transformation& tf, bool interpolate);
		~RockOdometry();

		void setGravityReference(const Direction& ref) {mGravityReference = ref;}
		void setGravityCovarianceScale(double scale) {mGravityCovarianceScale = scale;}

		/**
		 * @brief Set an anisotropic covariance model for the odometry SE3 edges.
		 * @param diagonal base covariance diagonal [tx,ty,tz,rx,ry,rz]; empty keeps the
		 *        legacy isotropic 100
		 * @param perMeter additive covariance diagonal per meter of edge translation
		 */
		void setOdometryCovariance(const std::vector<double>& diagonal,
		                           const std::vector<double>& perMeter);
		void handleNewVertex(IdType vertex);

		Transform getPose(timeval stamp);
		Transform getPose(base::Time t);
		Covariance<6> calculateCovariance(const Transform &tf);

	private:
		transformer::Transformation& mTransformation;
		Transform mLastOdometricPose;
		IdType mLastVertex;
		Direction mGravityReference;
		Solver* mSolver;
		bool mInterpolate;
		double mGravityCovarianceScale;
		bool mUseCovarianceModel;
		Eigen::Matrix<double,6,1> mCovDiagonal;
		Eigen::Matrix<double,6,1> mCovPerMeter;
	};
}

#endif
