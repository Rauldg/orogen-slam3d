#include "PointcloudMapper.hpp"
#include "RockOdometry.hpp"
#include "BaseLogger.hpp"
#include "Common.hpp"

#include <base/samples/Pointcloud.hpp>
#include <base/samples/RigidBodyState.hpp>

#include <slam3d/graph/boost/BoostGraph.hpp>
#include <slam3d/core/FileLogger.hpp>
#include <slam3d/solver/g2o/G2oSolver.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <cstdlib>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

using namespace slam3d;

PointcloudMapper::PointcloudMapper(std::string const& name)
    : PointcloudMapperBase(name)
{
}

PointcloudMapper::PointcloudMapper(std::string const& name, RTT::ExecutionEngine* engine)
    : PointcloudMapperBase(name, engine)
{
}

PointcloudMapper::~PointcloudMapper()
{
}

bool PointcloudMapper::optimize()
{
	mLogger->message(INFO, "Requested global optimization.");
	try
	{
		boost::unique_lock<boost::shared_mutex> guard(mGraphMutex);
		if(mGraph->optimize())
		{
			return true;
		}
	}catch (boost::lock_error &e)
	{
		mLogger->message(WARNING, "Could not access the pose graph for optimization! Is another operation still running?");
	}
	return false;
}

bool PointcloudMapper::generate_cloud()
{
	if(mPointcloudInProgress)
	{
		mLogger->message(WARNING, "Cloud generation already in progress.");
		return false;
	}else
	{
		// Publish accumulated cloud
		mLogger->message(INFO, "Requested pointcloud generation.");
		VertexObjectList vertices = mGraph->getVertices({mPclSensor->getName()});
		boost::thread projThread(&PointcloudMapper::sendPointcloud, this, vertices);
		return true;
	}
}

bool PointcloudMapper::generate_map()
{
	mLogger->message(INFO, "Requested map generation.");
	if(mMapInProgress)
	{
		mLogger->message(WARNING, "Map generation already in progress.");
		return false;
	}else
	{
		mMapInProgress = true;
		if(mGraph->optimized())
		{
			VertexObjectList vertices = mGraph->getVertices({mPclSensor->getName()});
			mMapInProgress = true;
			boost::thread projThread(&PointcloudMapper::rebuildMap, this, vertices);
		}else
		{
			sendMap();
		}
	}
	return true;
}

bool PointcloudMapper::force_add()
{
	mForceAdd = true;
	return true;
}

bool PointcloudMapper::write_envire()
{
	boost::filesystem::path env_path(_envire_path.value());
	boost::filesystem::create_directories(env_path);
	env_path += "/mls_map-";
	env_path += base::Time::now().toString(base::Time::Seconds, "%Y%m%d-%H%M");
	env_path += ".bin";

	std::ofstream ostream;
	ostream.open(env_path.string());
	if(!ostream.is_open())
	{
		mLogger->message(ERROR, (boost::format("Failed to serialize MLS map! Couldn't open file %1%") % env_path.string()).str());
		return false;
	}

	boost::archive::binary_oarchive bin_out(ostream);
	bin_out << mMultiLayerMap;
	ostream.close();
	return true;
}

bool PointcloudMapper::write_graph()
{
	mGraph->writeGraphToFile("slam3d_graph");
	return true;
}

bool PointcloudMapper::write_ply(const std::string& folder)
{
	mLogger->message(INFO, "Write pointcloud to PLY file.");
	VertexObjectList vertices = mGraph->getVertices({mPclSensor->getName()});
	PointCloud::Ptr accCloud;
	try
	{
		accCloud = buildPointcloud(vertices);
	}
	catch (boost::lock_error &e)
	{
		mLogger->message(ERROR, "Could not access the pose graph to build Pointcloud!");
		return false;
	}

	boost::filesystem::path ply_path(folder);
	boost::filesystem::create_directories(ply_path);
	ply_path += "pointcloud-";
	ply_path += base::Time::now().toString(base::Time::Seconds, "%Y%m%d-%H%M");
	ply_path += ".ply";

	pcl::PLYWriter ply_writer;
	return ply_writer.write(ply_path.string(), *accCloud) >= 0;
}

PointCloud::Ptr PointcloudMapper::buildPointcloud(const VertexObjectList& vertices)
{
	timeval start = mClock->now();
	boost::shared_lock<boost::shared_mutex> guard(mGraphMutex);
	PointCloud::Ptr cloud = mPclSensor->buildMap(vertices);
	timeval finish = mClock->now();
	int duration = finish.tv_sec - start.tv_sec;
	mLogger->message(INFO, (boost::format("Generated Pointcloud from %1% scans in %2% seconds.") % vertices.size() % duration).str());
	return cloud;
}

void PointcloudMapper::sendPointcloud(const VertexObjectList& vertices)
{
	PointCloud::Ptr accCloud;
	try
	{
		accCloud = buildPointcloud(vertices);
	}catch (boost::lock_error &e)
	{
		mLogger->message(ERROR, "Could not access the pose graph to build Pointcloud!");
		mPointcloudInProgress = false;
		return;
	}
	
	base::samples::Pointcloud mapCloud;
	createFromPcl(accCloud, mapCloud);
	_cloud.write(mapCloud);
	mPointcloudInProgress = false;
}

void PointcloudMapper::handleNewScan(const VertexObject& scan)
{
	addScanToMap(castToPointcloud(mMapper->getGraph()->getMeasurement(scan.measurementUuid)), scan.correctedPose);
}

void PointcloudMapper::addScanToMap(PointCloudMeasurement::Ptr scan, const Transform& pose)
{	
	PointCloud::ConstPtr pcl = scan->getPointCloud();
	boost::unique_lock<boost::shared_mutex> guard(mMapMutex);
	mMultiLayerMap.mergePointCloud(*pcl, pose * scan->getSensorPose());
}

void PointcloudMapper::clearMap()
{
	boost::unique_lock<boost::shared_mutex> guard(mMapMutex);
	mMultiLayerMap.clear();
}

void PointcloudMapper::rebuildMap(const VertexObjectList& vertices)
{
	timeval start = mClock->now();
	clearMap();

	boost::shared_lock<boost::shared_mutex> guard(mGraphMutex);
	for(VertexObjectList::const_iterator v = vertices.begin(); v != vertices.end(); ++v)
	{
		addScanToMap(castToPointcloud(mMapper->getGraph()->getMeasurement(v->measurementUuid)), v->correctedPose);
	}
	timeval finish = mClock->now();
	int duration = finish.tv_sec - start.tv_sec;
	mLogger->message(INFO, (boost::format("Completely rebuild map from %1% scans in %2% seconds.") % vertices.size() % duration).str());

	sendMap();
}

void PointcloudMapper::sendMap()
{
	// Publish the MLS-Map
	_mls.write(mMultiLayerMap);
	mMapInProgress = false;
	mScansReceived = 0;
}

bool PointcloudMapper::setLog_level(boost::int32_t value)
{
	switch(value)
	{
	case 4:
		mLogger->setLogLevel(FATAL);
		break;
	case 3:
		mLogger->setLogLevel(ERROR);
		break;
	case 2:
		mLogger->setLogLevel(WARNING);
		break;
	case 0:
		mLogger->setLogLevel(DEBUG);
		break;
	default:
		mLogger->setLogLevel(INFO);
	}
	return true;
}

bool PointcloudMapper::configureHook()
{	
	if (! PointcloudMapperBase::configureHook())
		return false;
		
	mClock = new Clock();
	switch(_log_type)
	{
	case 0:
		mLogger = new Logger(*mClock);
		break;
	case 1:
		mLogger = new BaseLogger();
		break;
	case 2:
		mLogger = new FileLogger(*mClock, "slam3d.log");
		break;
	default:
		mLogger = new Logger(*mClock);
		mLogger->message(WARNING, "Invalid logger type, using standard logger.");
	}

	setLog_level(_log_level);

	// Create all internal objects
	mSolver = new G2oSolver(mLogger);
	mPatchSolver = new G2oSolver(mLogger);

	mPclSensor = new PointCloudSensor("LaserScanner", mLogger);
	mPclSensor->setPatchSolver(mPatchSolver);

	mStorage = new MeasurementStorage;

	mGraph = new BoostGraph(mLogger, mStorage);
	mGraph->setSolver(mSolver);
	mGraph->fixNext();

	Transform startPose;
	if(_use_odometry_heading || _use_odometry_pose)
	{
		mStartPoseInitialized = false;
	}else
	{
		Eigen::Affine3d affine = _start_pose.get().toTransform();
		startPose.linear() = affine.linear();
		startPose.translation() = affine.translation();
		mStartPoseInitialized = true;
	}

	mMapper = new Mapper(mGraph, mLogger, startPose);
	mMapper->registerSensor(mPclSensor);

	// Read and set parameters
	mLogger->message(INFO, "=== PointCloudSensor - Parameters ===");
	mPclSensor->setMinPoseDistance(_min_translation, _min_rotation);
	mPclSensor->setNeighborRadius(_neighbor_radius, _max_neighbor_links);
	mPclSensor->setPatchBuildingRange(_patch_building_range);
	mPclSensor->setMapResolution(_map_resolution);
	mPclSensor->setMapOutlierRemoval(_map_outlier_radius, _map_outlier_neighbors);
	mPclSensor->setLinkPrevious(_sequential_icp);	
	mPclSensor->setMinLoopLength(_min_loop_length);
	mPclSensor->setRegistrationParameters(_gicp_config, false);
	mPclSensor->setRegistrationParameters(_gicp_coarse_config, true);

	mLogger->message(INFO, " === Mapper - Parameters ===");
	mLogger->message(INFO, (boost::format("use_odometry:           %1%") % _use_odometry.get()).str());	
	if(_use_odometry.get())
	{
		mOdometry = new RockOdometry("RockOdometry", mGraph, mSolver, mLogger, _robot2odometry, true);
		_robot2odometry.registerUpdateCallback(boost::bind(&PointcloudMapper::transformerCallback, this, _1));
		mLogger->message(INFO, (boost::format("add_odometry_edges:     %1%") % _add_odometry_edges).str());
		if(_add_odometry_edges)
		{
			mMapper->registerPoseSensor(mOdometry);
		}
		base::Vector3d vec3d = _gravity_reference.get();
		if(vec3d.norm() > 0)
		{
			mOdometry->setGravityReference(vec3d);
		}
	}else
	{
		mOdometry = NULL;
	}
	
	mLogger->message(INFO, (boost::format("scan_resolution:        %1%") % _scan_resolution).str());	
	mLogger->message(INFO, (boost::format("map_publish_rate:       %1%") % _map_publish_rate).str());
	mLogger->message(INFO, (boost::format("optimization_rate:      %1%") % _optimization_rate).str());
	
	mLogger->message(INFO, (boost::format("robot_name:             %1%") % _robot_name.get()).str());
	mLogger->message(INFO, (boost::format("laser_frame:            %1%") % _laser_frame.get()).str());
	mLogger->message(INFO, (boost::format("robot_frame:            %1%") % _robot_frame.get()).str());
	mLogger->message(INFO, (boost::format("odometry_frame:         %1%") % _odometry_frame.get()).str());
	mLogger->message(INFO, (boost::format("map_frame:              %1%") % _map_frame.get()).str());
	
	mScansAdded = 0;
	mScansReceived = 0;
	mForceAdd = false;
	mMapInProgress = false;
	mPointcloudInProgress = false;
	
	// Initialize MLS-Map
	mGridConf = _grid_config.get();
	mLogger->message(INFO, " = Grid - Parameters =");
	mLogger->message(INFO, (boost::format("min_x: %1%") % mGridConf.min_x).str());
	mLogger->message(INFO, (boost::format("max_x: %1%") % mGridConf.max_x).str());
	mLogger->message(INFO, (boost::format("min_y: %1%") % mGridConf.min_y).str());
	mLogger->message(INFO, (boost::format("max_y: %1%") % mGridConf.max_y).str());
	mLogger->message(INFO, (boost::format("min_z: %1%") % mGridConf.min_z).str());
	mLogger->message(INFO, (boost::format("max_z: %1%") % mGridConf.max_z).str());
	mLogger->message(INFO, (boost::format("resolution: %1%") % mGridConf.resolution).str());
	
	size_t x_size = (mGridConf.max_x - mGridConf.min_x) / mGridConf.resolution;
	size_t y_size = (mGridConf.max_y - mGridConf.min_y) / mGridConf.resolution;
	
	mMultiLayerMap = maps::grid::MLSMapSloped(maps::grid::Vector2ui(x_size, y_size), Eigen::Vector2d(mGridConf.resolution, mGridConf.resolution), _grid_mls_config.value());
	mMultiLayerMap.getId() = "/slam3d-mls";
	mMultiLayerMap.translate(Eigen::Vector3d(mGridConf.min_x, mGridConf.min_y, 0));

	// load a-priori map file
	if(!_apriori_ply_map.value().empty())
	{
		mPclSensor->loadPLY(_apriori_ply_map.value(), _robot_name.value());
		mScansAdded++;
		const VertexObject& ply_vertex = mGraph->getVertex(mPclSensor->getLastVertexId());
		addScanToMap(castToPointcloud(
			mStorage->get(ply_vertex.measurementUuid)),
			ply_vertex.correctedPose);
	}
	
	return true;
}

bool PointcloudMapper::startHook()
{
	if(!PointcloudMapperBase::startHook())
		return false;
	return true;
}


void PointcloudMapper::transformerCallback(const base::Time &time)
{
	mCurrentTime = time;
}


namespace
{
	// Wall-clock timer that reports on scope exit, so every return path of the callback is covered.
	struct ScanTimer
	{
		std::chrono::steady_clock::time_point start;
		std::function<void(double)> report;
		explicit ScanTimer(std::function<void(double)> r) : start(std::chrono::steady_clock::now()), report(r) {}
		~ScanTimer()
		{
			double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			report(ms);
		}
	};
}

void PointcloudMapper::scanTransformerCallback(const base::Time &ts, const base::samples::Pointcloud &scan_sample)
{
	// SLAM3D_SCAN_TIMING=1: log duration vs. vertex count; =2: also log edge count (copies the edge list, slows down the run).
	static const char* timing_env = getenv("SLAM3D_SCAN_TIMING");
	static const int timing_level = timing_env ? atoi(timing_env) : 0;
	const int scans_before = mScansAdded;
	bool did_optimize = false;
	bool did_mapgen = false;
	std::unique_ptr<ScanTimer> timer;
	if(timing_level > 0)
	{
		timer.reset(new ScanTimer([&](double ms)
		{
			if(mScansAdded == scans_before)
				return; // scan was skipped or rejected
			std::string edges = "-1";
			if(timing_level > 1)
			{
				try { edges = std::to_string(mGraph->getEdges({mPclSensor->getName()}).size()); }
				catch(std::exception&) {}
			}
			mLogger->message(INFO, (boost::format("SCANTIMING vertices=%1% edges=%2% ms=%3% optimized=%4% mapgen=%5%")
				% mScansAdded % edges % ms % did_optimize % did_mapgen).str());
		}));
	}

	if(ts != scan_sample.time)
	{
		mLogger->message(ERROR, "Timestamps of scan and transformer do not match.");
		return;
	}
	
	// Check if we have to set start pose from odometry
	if(!mStartPoseInitialized)
	{
		Transform startPose = Transform::Identity();
		Eigen::Affine3d affine = Eigen::Affine3d::Identity();
		try
		{
			if(!_robot2odometry.get(ts, affine, true) || !affine.matrix().allFinite())
			{
				mLogger->message(ERROR, (boost::format("Failed to receive a valid transform from '%1%' to '%2%'!")
					% _robot_frame.get() % _odometry_frame.get()).str());
				return;
			}
		}
		catch(std::exception &e)
		{
			mLogger->message(ERROR, e.what());
			return;
		}
		startPose.linear() = affine.linear();
		if(_use_odometry_pose)
		{
			startPose.translation() = affine.translation();
		}
		mMapper->setStartPose(startPose);
		mStartPoseInitialized = true;
	}

	if(scan_sample.points.size() == 0)
	{
		mLogger->message(ERROR, "Scan sample has no points.");
		return;
	}

	// Get laser pose
	Transform laserPose = Transform::Identity();
	try
	{
		Eigen::Affine3d affine;
		if(!_laser2robot.get(scan_sample.time, affine, true) || !affine.matrix().allFinite())
		{
			mLogger->message(ERROR, (boost::format("Failed to receive a valid transform from '%1%' to '%2%'!")
				% _laser_frame.get() % _robot_frame.get()).str());
			return;
		}
		laserPose.linear() = affine.linear();
		laserPose.translation() = affine.translation();
	}
	catch(std::exception &e)
	{
		mLogger->message(ERROR, e.what());
		return;
	}

	// Transform base::samples::Pointcloud --> Pointcloud
	PointCloud::Ptr cloud = createFromRockMessage(scan_sample);
	if(mScansAdded == 0 && _initial_patch_radius > 0)
	{
		mPclSensor->fillGroundPlane(cloud, _initial_patch_radius);
	}
	
	// Downsample and add to map
	PointCloudMeasurement::Ptr measurement;
	try
	{
		if(mOdometry)
		{
			if(!mPclSensor->checkMeasurementDistance(mOdometry->getPose(scan_sample.time)))
				return;
		}

		if(_scan_resolution > 0)
		{
			PointCloud::Ptr downsampled_cloud = mPclSensor->downsample(cloud, _scan_resolution);
			mLogger->message(DEBUG, (boost::format("Downsampled cloud has %1% points.") % downsampled_cloud->size()).str());
			measurement = PointCloudMeasurement::Ptr(new PointCloudMeasurement(downsampled_cloud, _robot_name.get(), mPclSensor->getName(), laserPose));
		}else
		{
			measurement = PointCloudMeasurement::Ptr(new PointCloudMeasurement(cloud, _robot_name.get(), mPclSensor->getName(), laserPose));
		}
		
		mScansReceived++;

		bool added = false;
		if(mOdometry)
		{
			added = mPclSensor->addMeasurement(measurement, mOdometry->getPose(measurement->getTimestamp()));
			if(added)
			{
				mCurrentDrift = orthogonalize(mMapper->getCurrentPose() * mOdometry->getPose(measurement->getTimestamp()).inverse());
			}
		}else
		{
			added = mPclSensor->addMeasurement(measurement);
		}

		if(added)
		{
			mScansAdded++;
			mForceAdd = false;

			mPclSensor->linkLastToNeighbors();

			if((int)mGraph->getNumOfNewConstraints() >= _optimization_rate)
			{
				did_optimize = true;
				optimize();
			}

			// Merge after optimization, so the keyframe enters the map at its corrected pose.
			// getVertex returns by value, so this reads the post-optimization pose.
			handleNewScan(mGraph->getVertex(mPclSensor->getLastVertexId()));

			if(_map_publish_rate > 0 && (mScansAdded % _map_publish_rate) == 0)
			{
				did_mapgen = true;
				generate_map();
			}
		}else
		{
			if((_map_update_rate > 0) && (mScansReceived >= _map_update_rate) && !mMapInProgress)
			{
				addScanToMap(measurement, mMapper->getCurrentPose());
				sendMap();
			}
		}
		mCurrentTime = scan_sample.time;
	}catch(std::exception& e)
	{
		mLogger->message(ERROR, (boost::format("Adding scan to map failed: %1%") % e.what()).str());
	}
}

void PointcloudMapper::updateHook()
{
	try
	{
		PointcloudMapperBase::updateHook();

		// Check if we already received data and have a valid time
		if(mCurrentTime.isNull())
		{
			mLogger->message(WARNING, "Current time is null, not sending transforms.");
			return;
		}

		// Send the calculated transforms
		base::samples::RigidBodyState rbs;
		rbs.invalidateCovariances();
		rbs.sourceFrame = _robot_frame.get();
		rbs.targetFrame = _map_frame.get();
		rbs.time = mCurrentTime;
		
		if(mOdometry)
		{
			rbs.setTransform(mCurrentDrift * mOdometry->getPose(mCurrentTime));
			_robot2map.write(rbs);

			rbs.sourceFrame = _odometry_frame.get();
			rbs.setTransform(mCurrentDrift);
			_odometry2map.write(rbs);
		}else
		{
			rbs.setTransform(mPclSensor->getCurrentPose());
			_robot2map.write(rbs);
		}
	}catch(std::exception &e)
	{
		mLogger->message(ERROR, e.what());
		return;
	}
}

void PointcloudMapper::errorHook()
{
	PointcloudMapperBase::errorHook();
}

void PointcloudMapper::stopHook()
{
	PointcloudMapperBase::stopHook();
}

void PointcloudMapper::cleanupHook()
{
	PointcloudMapperBase::cleanupHook();

	// Unregister the transformer callback
	_robot2odometry.registerUpdateCallback(boost::function<void (const base::Time &ts)>());

	delete mMapper;
	delete mGraph;
	delete mPclSensor;
	if(mOdometry)
		delete mOdometry;
	delete mSolver;
	delete mPatchSolver;
	delete mStorage;
	delete mLogger;
	delete mClock;
}
