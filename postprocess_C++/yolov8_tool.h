#ifndef _YOLOV8_TOOL_H_
#define _YOLOV8_TOOL_H_

#include <iostream>
#include "metadata.h"
#include "math_utils.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <algorithm>


struct DetectionYolov8 {
    BBox bbox;
    int track_id;
    int label;
    float class_prob;
};

std::vector<DetectionYolov8> nms(std::vector<DetectionYolov8>& dets, float thresh_iou);
std::vector<DetectionYolov8> hard_nms(std::vector<DetectionYolov8>& dets, float thresh_iou, int top_k, int candidate_size);

class Yolov8Tool {
    private:
        int model_w;
        int model_h;
        int num_reg;
        int num_class;
        float thresh_pre_nms;
        float thresh_nms;

        void decode(float* cls, float* bb,int stride, std::vector<DetectionYolov8>& dets);
        std::vector<std::vector<float>> dist2bbox(std::vector<std::vector<float>> distance, std::vector<std::vector<float>> anchor_points, bool xywh = true, int dim = -1);
        std::vector<std::vector<float>> make_anchor(int h, int w, float grid_cell_offset = 0.5);
        std::vector<std::vector<float>> dfl(float* bb,int h, int w);
        std::vector<float> dfl_target(float* bb, int target_idx);
        std::vector<float> dist2bbox_target(std::vector<float> bbox, std::vector<float> anchor_points, bool xywh = true, int dim = -1);
        std::vector<float> conv;
    public:
        Yolov8Tool(int model_w, int model_h, int num_reg, int num_class, float thresh_pre_nms, float thresh_nms);
        ~Yolov8Tool();
        std::vector<DetectionYolov8> postprocess(float* cls_8, float* cls_16, float* cls_32, float* bb_8, float* bb_16, float* bb_32);     
        const std::vector<int> STRIDE_FPN {8, 16, 32};
        

};

#endif