/*
 * ICPOdometry.h
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#ifndef ICPODOMETRY_H_
#define ICPODOMETRY_H_

#include "Cuda/internal.h"

#include <vector>
#include <sophus/se3.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

class ICPOdometry
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        ICPOdometry(int width,
                     int height,
                     float cx, float cy, float fx, float fy,
                     float distThresh = 0.10f,
                     float angleThresh = sinf(20.f * 3.14159254f / 180.f));

        virtual ~ICPOdometry();

        void initICP(unsigned short * depth, const float depthCutoff = 20.0f);

        void initICPModel(unsigned short * depth, const float depthCutoff = 20.0f);

        void getIncrementalTransformation(Sophus::SE3d & T_prev_curr, int threads, int blocks);

        float lastError;
        float lastInliers;

    private:
        std::vector<DeviceArray2D<unsigned short>> depth_tmp_;

        std::vector<DeviceArray2D<float>> vmaps_prev_;
        std::vector<DeviceArray2D<float>> nmaps_prev_;

        std::vector<DeviceArray2D<float>> vmaps_curr_;
        std::vector<DeviceArray2D<float>> nmaps_curr_;

        CameraIntrinsic intr;

        DeviceArray<Eigen::Matrix<float,29,1,Eigen::DontAlign>> sumData_;
        DeviceArray<Eigen::Matrix<float,29,1,Eigen::DontAlign>> outData_;

        static const int NUM_PYRS = 3;

        std::vector<int> iterations_;

        float dist_thresh_;
        float angle_thresh_;

        int width_;
        int height_;
        float cx_, cy_, fx_, fy_;
};

#endif /* ICPODOMETRY_H_ */
