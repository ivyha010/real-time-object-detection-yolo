#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <memory>   
#include <vector>
#include <string>

int main() {
    // Open webcam
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open webcam\n";
        return -1;
    }

    // Load YOLO model 
    std::string modelPath = "yolo11n.onnx"; 
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    // use GPU (if available)
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Detection loop
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Preprocess frame
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);

        net.setInput(blob);

        // Run forward pass
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Parse detections
        for (auto& out : outputs) {
            for (int i = 0; i < out.rows; i++) {
                float confidence = out.at<float>(i, 4);
                if (confidence > 0.5) {
                    int x = static_cast<int>(out.at<float>(i, 0) * frame.cols);
                    int y = static_cast<int>(out.at<float>(i, 1) * frame.rows);
                    int w = static_cast<int>(out.at<float>(i, 2) * frame.cols);
                    int h = static_cast<int>(out.at<float>(i, 3) * frame.rows);

                    cv::Rect box(x - w/2, y - h/2, w, h);
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // Show detections
        cv::imshow("YOLO Detection", frame);

        // Exit on 'q'
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
