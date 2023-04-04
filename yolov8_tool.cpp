#include "yolov8_tool.h"

Yolov8Tool::~Yolov8Tool() 
{
    
}

Yolov8Tool::Yolov8Tool(int model_w, int model_h, int num_reg, int num_class, float thresh_pre_nms, float thresh_nms)
    : model_h(model_h), model_w(model_w), num_reg(num_reg), num_class(num_class), thresh_pre_nms(thresh_pre_nms), thresh_nms(thresh_nms)
{
    conv.resize(num_reg);
    for (int i = 0; i < num_reg; i++) {
        conv[i] = static_cast<float>(i);
    }
    
}

std::vector<DetectionYolov8> Yolov8Tool::postprocess(float* cls_8, float* cls_16, float* cls_32, float* bb_8, float* bb_16, float* bb_32)
{
    std::vector<DetectionYolov8> dets, new_dets;

    decode(cls_8, bb_8, 8, dets);
    decode(cls_16, bb_16, 16, dets);
    decode(cls_32, bb_32, 32, dets);

    printf("PreNMS: %d\n", dets.size());
    // dets = nms(dets, thresh_nms);
    dets = hard_nms(dets, thresh_nms, -1, 100);

    for (int i = 0; i < dets.size(); i++) {
        DetectionYolov8 det = dets[i];
        float w = det.bbox.x2 - det.bbox.x1;
        if (w > 1) continue;
        new_dets.push_back(det);
    }
    
    printf("AfterNMS: %d\n", new_dets.size());
    for (int i = 0; i < new_dets.size(); i++) {
        DetectionYolov8 det = new_dets[i];
        printf("x1: %f, y1: %f, x2: %f, y2: %f, class: %d, prob: %f \n", det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2, det.label, det.class_prob);
    }
    return new_dets;
}

std::vector<std::vector<float>> Yolov8Tool::make_anchor(int h, int w, float grid_cell_offset) {
    std::vector<std::vector<float>> anchor_points;
    std::vector<float> sx(w);
    for (int i = 0; i < w; i++) {
        sx[i] = static_cast<float>(i) + grid_cell_offset;
    }
    std::vector<float> sy(h);
    for (int i = 0; i < h; i++) {
        sy[i] = static_cast<float>(i) + grid_cell_offset;
    }
    std::vector<std::vector<float>> mesh(h, std::vector<float>(w));
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            mesh[i][j] = sy[i];
        }
    }
    for (int j = 0; j < w; j++) {
        for (int i = 0; i < h; i++) {
            mesh[i][j] = sx[j];
        }
    }
    std::vector<std::vector<float>> flat(h*w, std::vector<float>(2));
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            flat[i*w+j][0] = mesh[i][j];
            flat[i*w+j][1] = sy[i];
        }
    }
    std::swap(flat[0][0], flat[0][1]);
    std::swap(flat[1][0], flat[1][1]);
    return flat;
}

std::vector<std::vector<float>> Yolov8Tool::dist2bbox(std::vector<std::vector<float>> distance, std::vector<std::vector<float>> anchor_points, bool xywh, int dim) {
    //  lt is  2 first values of each row of distance
    std::vector<std::vector<float>> lt(distance.size(), std::vector<float>(2));
    std::vector<std::vector<float>> rb(distance.size(), std::vector<float>(2));
    for (int i = 0; i< distance.size(); i++) {
        for (int j = 0; j < 2; j++) {
            lt[i][j] = distance[i][j];
            rb[i][j] = distance[i][j+2];
        }
    }


    std::vector<std::vector<float>> x1y1(anchor_points.size(), std::vector<float>(2));
    std::vector<std::vector<float>> x2y2(anchor_points.size(), std::vector<float>(2));
    for (int i = 0; i < anchor_points.size(); i++) {
        for (int j = 0; j < 2; j++) {
            x1y1[i][j] = anchor_points[i][j] - lt[i][j];
            x2y2[i][j] = anchor_points[i][j] + rb[i][j];
        }
    }
    if (xywh) {
        std::vector<std::vector<float>> c_xy(anchor_points.size(), std::vector<float>(2));
        std::vector<std::vector<float>> wh(anchor_points.size(), std::vector<float>(2));
        for (int i = 0; i < anchor_points.size(); i++) {
            for (int j = 0; j < 2; j++) {
                c_xy[i][j] = (x1y1[i][j] + x2y2[i][j]) / 2;
                wh[i][j] = x2y2[i][j] - x1y1[i][j];
            }
        }
        std::vector<std::vector<float>> bbox(anchor_points.size(), std::vector<float>(4));
        for (int i = 0; i < anchor_points.size(); i++) {
            bbox[i][0] = c_xy[i][0];
            bbox[i][1] = c_xy[i][1];
            bbox[i][2] = wh[i][0];
            bbox[i][3] = wh[i][1];
        }
        return bbox;
    } else {
        std::vector<std::vector<float>> bbox(anchor_points.size(), std::vector<float>(4));
        for (int i = 0; i < anchor_points.size(); i++) {
            bbox[i][0] = x1y1[i][0];
            bbox[i][1] = x1y1[i][1];
            bbox[i][2] = x2y2[i][0];
            bbox[i][3] = x2y2[i][1];
        }
        return bbox;
    }
}
std::vector<float> Yolov8Tool::dist2bbox_target(std::vector<float> bbox, std::vector<float> anchor_points, bool xywh, int dim)
{
    std::vector<float> lt(2);
    std::vector<float> rb(2);
    for (int i = 0; i < 2; i++) {
        lt[i] = bbox[i];
        rb[i] = bbox[i+2];
    }
    std::vector<float> x1y1(2);
    std::vector<float> x2y2(2);
    for (int i = 0; i < 2; i++) {
        x1y1[i] = anchor_points[i] - lt[i];
        x2y2[i] = anchor_points[i] + rb[i];
    }
    if (xywh) {
        std::vector<float> c_xy(2);
        std::vector<float> wh(2);
        for (int i = 0; i < 2; i++) {
            c_xy[i] = (x1y1[i] + x2y2[i]) / 2;
            wh[i] = x2y2[i] - x1y1[i];
        }
        std::vector<float> bbox(4);
        for (int i = 0; i < 2; i++) {
            bbox[i] = c_xy[i];
            bbox[i+2] = wh[i];
        }
        return bbox;
    } else {
        std::vector<float> bbox(4);
        for (int i = 0; i < 2; i++) {
            bbox[i] = x1y1[i];
            bbox[i+2] = x2y2[i];
        }
        return bbox;
    }
}

std::vector<std::vector<float>> Yolov8Tool::dfl(float* bb, int h, int w)
{
    // std::vector<float>conv = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    int num_bb = h*w;
    std::vector<std::vector<float>> bboxes(num_bb, std::vector<float>(4));
    
    for (int i = 0; i < num_bb; i++) {
        for (int j = 0; j < 4; j++) {
            std::vector<float> cal(num_reg);
            float sum = 0;
            for (int k = 0; k < num_reg; k++) {
                cal[k] = bb[num_reg*4*i+num_reg*j+k];
            }
            //  do softmax for cal
            float sum_exp = 0;
            for (int k = 0; k < num_reg; k++) {
                sum_exp += exp(cal[k]);
            }
            for (int k = 0; k < num_reg; k++) {
                cal[k] = exp(cal[k]) / sum_exp;
            }
            for (int k = 0; k < num_reg; k++) {
                sum += (cal[k] * conv[k]);
            }
            bboxes[i][j] = sum;
        }
    }
    return bboxes;
}

std::vector<float> Yolov8Tool::dfl_target(float* bb, int target_idx)
{
    // std::vector<float>conv = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    std::vector<float> bbox(4);
    for (int j = 0; j < 4; j++) {
        std::vector<float> cal(num_reg);
        float sum = 0;
        for (int k = 0; k < num_reg; k++) {
            cal[k] = bb[num_reg*4*target_idx+num_reg*j+k];
        }
        //  do softmax for cal
        float sum_exp = 0;
        for (int k = 0; k < num_reg; k++) {
            sum_exp += exp(cal[k]);
        }
        for (int k = 0; k < num_reg; k++) {
            cal[k] = exp(cal[k]) / sum_exp;
        }
        for (int k = 0; k < num_reg; k++) {
            sum += (cal[k] * conv[k]);
        }
        bbox[j] = sum;
    }
    return bbox;
}


void Yolov8Tool::decode(float* cls, float* bb,int stride, std::vector<DetectionYolov8>& dets)
{
    int h = model_h / stride;
    int w = model_w / stride;
    std::vector<std::vector<float>> anchors = make_anchor(h, w);
    
    int num_bb = h*w;
    // int num_dets = 0;
    for (int i = 0; i < num_bb; i++) {
        float cls_max = 0;
        int cls_max_idx = 0;
        for (int j = 0; j < num_class; j++) {
            if (cls[num_class*i+j] > cls_max) {
                cls_max = cls[num_class*i+j];
                cls_max_idx = j;
            }
        }
        
        DetectionYolov8 det;
        if (cls_max > thresh_pre_nms) {

            det.label = cls_max_idx;
            det.class_prob = cls_max;
            std::vector<float> bbox_target = dfl_target(bb, i);
            bbox_target = dist2bbox_target(bbox_target, anchors[i], true, 1);

            float x = bbox_target[0]*stride/model_w;
            float y = bbox_target[1]*stride/model_h;
            float w = bbox_target[2]*stride/model_w;
            float h = bbox_target[3]*stride/model_h;

            det.bbox.x1 = x - w / 2;
            det.bbox.y1 = y - h / 2;
            det.bbox.x2 = x + w / 2;
            det.bbox.y2 = y + h / 2;
            dets.push_back(det);
            // num_dets++;
        }   
    }
    // std::cout << "num_dets = " << num_dets << std::endl;
}



std::vector<DetectionYolov8> nms(std::vector<DetectionYolov8> &outputs, float iou_threshold)
{
    std::vector<DetectionYolov8> selected_outputs;
    std::sort(outputs.begin(), outputs.end(), [](const DetectionYolov8& a, const DetectionYolov8& b) -> bool {
        return a.class_prob > b.class_prob;
    });
    for (unsigned int i = 0; i < outputs.size(); i++) {
        bool non_overlap = true;
        for (unsigned int j = 0; j < selected_outputs.size(); j++) {
            float iou_score = math_utils::iou(selected_outputs[j].bbox, outputs[i].bbox);                        
            if (iou_score > iou_threshold) {
                non_overlap = false;
                break;
            }
        }
        if (non_overlap)
            selected_outputs.push_back(outputs[i]);
    }
    return selected_outputs;
}

std::vector<DetectionYolov8> hard_nms(std::vector<DetectionYolov8>& box_scores, float iou_threshold, int top_k = -1, int candidate_size = 400) {
    
    std::vector<DetectionYolov8> selected_boxes;
    std::vector<int> picked;
    std::sort(box_scores.begin(), box_scores.end(), [](const DetectionYolov8& a, const DetectionYolov8& b) -> bool {
        return a.class_prob > b.class_prob;
    });
    int size = std::min(static_cast<int>(box_scores.size()), candidate_size);
    for (int i = 0; i < size; ++i) {
        if (box_scores[i].class_prob == 0) {
            continue;
        }
        picked.push_back(i);
        if (top_k > 0 && picked.size() == top_k) {
            break;
        }
        for (int j = i + 1; j < size; ++j) {
            if (box_scores[j].class_prob == 0) {
                continue;
            }
            if (math_utils::iou(box_scores[picked.back()].bbox, box_scores[j].bbox) > iou_threshold) {
                box_scores[j].class_prob = 0;
            }
            
        }
        std::sort(box_scores.begin() + i + 1, box_scores.end(), [](const DetectionYolov8& a, const DetectionYolov8& b) { return a.class_prob > b.class_prob; });
       
    }

    for (auto it = picked.begin(); it != picked.end(); ++it) {
        selected_boxes.push_back(box_scores[*it]);
    }

    return selected_boxes;

}