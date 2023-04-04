# yolov8_postprocess_for_cut_model
Postprocessing for new sota object detection model yolov8. The models here is yolov8 cut version (cut all last postprocess layers)

When deploying yolov8 model to many Ai devices like ai edge devices, SoCs,... The full version of yolov8 is not compatible because it contains some postprocess layers. When removes those layers, the postprocess published by yolov8 author is not available anymore. 

The python code is postproces code in Python with provided model yolov8n_cut.onnx

The C++ code is postprocess code for SoCs device (here is Qualcomm SoCs), the provided onnx modified_yolov8n_cut.onnx will be use to convert to the compatible form with the SoCs. 

Note: For each SoCs, the output of model process is different, that makes the postprocess different.
