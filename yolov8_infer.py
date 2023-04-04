import numpy as np
import onnxruntime as rt
import cv2
import onnx 



def preprocess(img, input_shape):
    # Resize and normalize
    img = cv2.resize(img, input_shape)
    img = img.astype(np.float32)
    img /= 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # transpose to BCHW
    img = np.transpose(img, (0, 3, 1, 2))
    return img


def run_model(img, model_path):
    # Load model
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    # Run model
    output = sess.run(None, {input_name: img})
    return output

def make_anchors(h_list,w_list, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensors = [], []
    for i, stride in enumerate(strides):
        sx = np.arange(w_list[i], dtype=np.float32) + grid_cell_offset
        sy = np.arange(h_list[i], dtype=np.float32) + grid_cell_offset
        sy, sx = np.meshgrid(sy, sx)
        anchor_points.append(np.stack((sx, sy), axis=-1).reshape(-1,2))
        stride_tensors.append(np.full((w_list[i] * h_list[i], 1), stride, dtype=np.float32))
    anchors = np.concatenate(anchor_points)
    strides = np.concatenate(stride_tensors)
    anchors = anchors.transpose((1,0))
    strides = strides.transpose((1,0))
    #  swap anchors[0] and anchors[1]
    anchors[[0,1]]= anchors[[1,0]]

    anchors = np.expand_dims(anchors, axis=0)
    return anchors, strides

def make_anchor(h,w,grid_cell_offset=0.5):
    anchor_points = []
    sx = np.arange(w, dtype=np.float32) + grid_cell_offset
    sy = np.arange(h, dtype=np.float32) + grid_cell_offset
    sy, sx = np.meshgrid(sy, sx)
    anchor_points.append(np.stack((sx, sy), axis=-1).reshape(-1,2))
    anchors = np.concatenate(anchor_points)
    anchors = anchors.transpose((1,0))
    #  swap anchors[0] and anchors[1]
    anchors[[0,1]]= anchors[[1,0]]
    anchors = np.expand_dims(anchors, axis=0)
    return anchors


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), axis=dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), axis=dim)  # xyxy bbox

def dfl(bboxes):
    conv = np.expand_dims(np.arange(16, dtype=np.float32), axis=(1, 2, 3))
    conv = conv.transpose((1,0,2,3))
    b,c,a = bboxes.shape
    bboxes_reshape = np.transpose(bboxes.reshape((b, 4, conv.shape[1], a)), (0, 2, 1, 3))
    bboxes_softmax = np.apply_along_axis(lambda arr: np.exp(arr) / np.sum(np.exp(arr)), 1, bboxes_reshape)
    conv_res = np.sum(bboxes_softmax * conv, axis=1)

    return conv_res


def postprocess(score,bboxes,anchors,strides):
    boxes = dfl(bboxes)
    bbox = dist2bbox(boxes, anchors, xywh=True, dim=1)
    print(bbox)
    bbox = bbox * strides
    print(bbox)
    

    # do sigmoid for score
    score = 1 / (1 + np.exp(-score))


    bbox = bbox[0].transpose((1,0))
    score = score[0].transpose((1,0))

    dets = []
    model_w = 320
    model_h = 320



    num_cls = 0

    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            if score[i][j] > 0.5:
                num_cls += 1
    print(num_cls)            

    for i in range(score.shape[0]):
        max_score = np.max(score[i])
        max_score_index = np.argmax(score[i])
        if max_score > 0.5:
            dets.append([bbox[i][0]/model_w,bbox[i][1]/model_h,bbox[i][2]/model_w,bbox[i][3]/model_h, max_score,max_score_index])
            # print("x",bbox[i][0]/model_w,"y",bbox[i][1]/model_h,"w",bbox[i][2]/model_w,"h",bbox[i][3]/model_h,"score",max_score)
    return dets

        

def cacl_intersection_area(box1, box2):
    b1_x1 = box1[0]-box1[2]/2
    b1_y1 = box1[1]-box1[3]/2
    b1_x2 = box1[0]+box1[2]/2
    b1_y2 = box1[1]+box1[3]/2

    b2_x1 = box2[0]-box2[2]/2
    b2_y1 = box2[1]-box2[3]/2
    b2_x2 = box2[0]+box2[2]/2
    b2_y2 = box2[1]+box2[3]/2

    intersection_w = min(b1_x2, b2_x2) - max(b1_x1, b2_x1)
    intersection_w = max(intersection_w, 0.0)
    intersection_h = min(b1_y2, b2_y2) - max(b1_y1, b2_y1)
    intersection_h = max(intersection_h, 0.0)
    intersection_area = max(intersection_h * intersection_w, 0.0)
    return intersection_area

def iou(box1, box2):
    w1 = box1[2]
    h1 = box1[3]
    w2 = box2[2]
    h2 = box2[3]

    if w1 <= 0 and h1 <= 0 and w2 <= 0 and h2 <= 0:
        return 0

    intersection_area = cacl_intersection_area(box1, box2)

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area

def nms(dets, thresh):
    selected_outputs = []
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    for i in range(len(dets)):
        non_overlap = True
        for j in range(len(selected_outputs)):
            iou_score = iou(selected_outputs[j], dets[i])
            if iou_score > thresh:
                non_overlap = False
                break
        if non_overlap:
            selected_outputs.append(dets[i])
    return selected_outputs








def main():
    # Load image
    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    img = preprocess(img, (320, 320))
    print(img.shape)

    # Run model
    pred_onnx = run_model(img, 'yolov8n_cut.onnx')


    score_8 = pred_onnx[1]
    bbox_8 = pred_onnx[0]
    score_16 = pred_onnx[3]
    bbox_16 = pred_onnx[2]
    score_32 = pred_onnx[5]
    bbox_32 = pred_onnx[4]

    score_8 = np.reshape(score_8, (1, 80, -1))
    score_16 = np.reshape(score_16, (1, 80, -1))
    score_32 = np.reshape(score_32, (1, 80, -1))

    bbox_8 = np.reshape(bbox_8, (1, 64, -1))
    bbox_16 = np.reshape(bbox_16, (1, 64, -1))
    bbox_32 = np.reshape(bbox_32, (1, 64, -1))

    num_score_8 = 0
    num_score_16 = 0
    num_score_32 = 0

    for i in range(score_8.shape[2]):
        for j in range(score_8.shape[1]):
            if (1/(1+np.exp(-score_8[0][j][i]))) > 0.5:
                num_score_8 += 1
    print(num_score_8)

    for i in range(score_16.shape[2]):
        for j in range(score_16.shape[1]):
            if (1/(1+np.exp(-score_16[0][j][i]))) > 0.5:
                num_score_16 += 1
    print(num_score_16)

    for i in range(score_32.shape[2]):
        for j in range(score_32.shape[1]):
            if (1/(1+np.exp(-score_32[0][j][i]))) > 0.5:
                num_score_32 += 1
    print(num_score_32)

    score = np.concatenate((score_8, score_16, score_32), axis=2)
    bbox = np.concatenate((bbox_8, bbox_16, bbox_32), axis=2)


    stride = [8,16,32]
    h_list = [pred_onnx[1].shape[2],pred_onnx[3].shape[2],pred_onnx[5].shape[2]]
    w_list = [pred_onnx[1].shape[3],pred_onnx[3].shape[3],pred_onnx[5].shape[3]]
    x = make_anchors(h_list,w_list,stride,0.5)
    anchors = x[0]
    strides = x[1]



    dets = postprocess(score,bbox,anchors,strides)
    dets = np.array(dets)
    print(dets.shape)
    dets = nms(dets, 0.4)
    img_show = cv2.imread('test.jpg')
    img_w = img_show.shape[1]
    img_h = img_show.shape[0]

    print(len(dets))
    for det in dets:
        score = det[4]
        class_index = det[5]
        x,y,w,h = det[0],det[1],det[2],det[3]
        x1 = (x - w/2)*img_w
        y1 = (y - h/2)*img_h
        w = w*img_w
        h = h*img_h

        # draw bbox
        cv2.rectangle(img_show, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (0, 255, 0), 2)
        # draw class
        # cv2.putText(img_show, str(class_index), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # draw score
        # cv2.putText(img_show, str(score), (int(x1), int(y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # save image
    cv2.imwrite('test_out_3.jpg', img_show)

    




    



# Run main
if __name__ == '__main__':
    main()
