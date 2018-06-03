/*
 * ICPOdometry.cpp
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#include "ICPOdometry.h"

ICPOdometry::ICPOdometry(int width,
                          int height,
                          float cx, float cy, float fx, float fy,
                          float distThresh,
                          float angleThresh)
: lastError(0),
  lastInliers(width * height),
  dist_thresh_(distThresh),
  angle_thresh_(angleThresh),
  width_(width),
  height_(height),
  cx_(cx), cy_(cy), fx_(fx), fy_(fy)
{
    sumData_.create(MAX_THREADS);
    outData_.create(1);

    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations_.reserve(NUM_PYRS);

    depth_tmp_.resize(NUM_PYRS);

    vmaps_prev_.resize(NUM_PYRS);
    nmaps_prev_.resize(NUM_PYRS);

    vmaps_curr_.resize(NUM_PYRS);
    nmaps_curr_.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i)
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp_[i].create(pyr_rows, pyr_cols);

        vmaps_prev_[i].create(pyr_rows*3, pyr_cols);
        nmaps_prev_[i].create(pyr_rows*3, pyr_cols);

        vmaps_curr_[i].create(pyr_rows*3, pyr_cols);
        nmaps_curr_[i].create(pyr_rows*3, pyr_cols);
    }
}

ICPOdometry::~ICPOdometry()
{

}

void ICPOdometry::initICP(unsigned short * depth, const float depthCutoff)
{
    depth_tmp_[0].upload(depth, sizeof(unsigned short) * width_, height_, width_);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp_[i - 1], depth_tmp_[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp_[i], vmaps_curr_[i], depthCutoff);
        createNMap(vmaps_curr_[i], nmaps_curr_[i]);
    }

    cudaDeviceSynchronize();
}

void ICPOdometry::initICPModel(unsigned short * depth, const float depthCutoff)
{
    depth_tmp_[0].upload(depth, sizeof(unsigned short) * width_, height_, width_);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp_[i - 1], depth_tmp_[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp_[i], vmaps_prev_[i], depthCutoff);
        createNMap(vmaps_prev_[i], nmaps_prev_[i]);
    }

    cudaDeviceSynchronize();
}

void ICPOdometry::getIncrementalTransformation(Sophus::SE3d & T_prev_curr, int threads, int blocks)
{
    iterations_[0] = 10;
    iterations_[1] = 5;
    iterations_[2] = 4;

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        for(int j = 0; j < iterations_[i]; j++)
        {
            float residual_inliers[2];
            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            estimateStep(T_prev_curr.rotationMatrix().cast<float>().eval(),
                         T_prev_curr.translation().cast<float>().eval(),
                         vmaps_curr_[i],
                         nmaps_curr_[i],
                         intr(i),
                         vmaps_prev_[i],
                         nmaps_prev_[i],
                         dist_thresh_,
                         angle_thresh_,
                         sumData_,
                         outData_,
                         A_icp.data(),
                         b_icp.data(),
                         &residual_inliers[0],
                         threads,
                         blocks);

            lastError = sqrt(residual_inliers[0]) / residual_inliers[1];
            lastInliers = residual_inliers[1];

            const Eigen::Matrix<double, 6, 1> update = A_icp.cast<double>().ldlt().solve(b_icp.cast<double>());

            T_prev_curr = Sophus::SE3d::exp(update) * T_prev_curr;
        }
    }
}
