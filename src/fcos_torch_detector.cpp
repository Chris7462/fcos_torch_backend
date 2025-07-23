#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

// COCO class labels
const std::vector<std::string> COCO_CLASSES =
{
  "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
  "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
  "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
  "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
  "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

class FCOSDetector
{
public:
  FCOSDetector(const std::string& model_path, bool use_cuda = false)
    : device(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    try {
      // Load the TorchScript model
      model = torch::jit::load(model_path);
      model.to(device);
      model.eval();
      std::cout << "Model loaded successfully on " <<
          (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
    } catch (const std::exception& e) {
      throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
  }

  // Convert OpenCV Mat to PyTorch tensor
  torch::Tensor mat_to_tensor(const cv::Mat& image) {
    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, 1.0/255.0);

    // Convert from HWC to CHW and add batch dimension
    auto tensor = torch::from_blob(float_image.data, {image.rows, image.cols, 3}, torch::kFloat);
    tensor = tensor.permute({2, 0, 1}); // HWC to CHW
    tensor = tensor.unsqueeze(0); // Add batch dimension

      return tensor.to(device);
  }

  // Run inference
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predict(const cv::Mat& image) {
    torch::NoGradGuard no_grad;

    // Convert image to tensor (remove batch dimension since we'll create a list)
    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, 1.0/255.0);

    auto tensor = torch::from_blob(float_image.data, {image.rows, image.cols, 3}, torch::kFloat);
    tensor = tensor.permute({2, 0, 1}); // HWC to CHW
    tensor = tensor.to(device);

    // Create a list of tensors (this is what FCOS expects)
    std::vector<torch::Tensor> image_list = {tensor};

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image_list);

    auto output = model.forward(inputs);

    // FCOS returns a tuple: (Losses, Detections) in scripting mode
    auto output_tuple = output.toTuple();

    // Get the detections (second element of the tuple)
    auto detections = output_tuple->elements()[1];
    auto detection_list = detections.toList();
    auto detection_dict = detection_list.get(0).toGenericDict();

    // Extract predictions
    auto boxes = detection_dict.at("boxes").toTensor().to(torch::kCPU);
    auto scores = detection_dict.at("scores").toTensor().to(torch::kCPU);
    auto labels = detection_dict.at("labels").toTensor().to(torch::kCPU);

    return std::make_tuple(boxes, scores, labels);
  }

  // Draw predictions on image
  void draw_predictions(cv::Mat& image, const torch::Tensor& boxes, const torch::Tensor& scores, const torch::Tensor& labels, float confidence_threshold = 0.5f) {
    auto boxes_a = boxes.accessor<float, 2>();
    auto scores_a = scores.accessor<float, 1>();
    auto labels_a = labels.accessor<long, 1>();

    int num_detections = scores.size(0);

    for (int i = 0; i < num_detections; ++i) {
      float score = scores_a[i];

      if (score >= confidence_threshold) {
        // Get bounding box coordinates
        int x1 = static_cast<int>(boxes_a[i][0]);
        int y1 = static_cast<int>(boxes_a[i][1]);
        int x2 = static_cast<int>(boxes_a[i][2]);
        int y2 = static_cast<int>(boxes_a[i][3]);

        // Get class label
        size_t label_idx = labels_a[i];
        std::string class_name = (label_idx < COCO_CLASSES.size()) ? COCO_CLASSES[label_idx] : "unknown";

        // Draw bounding box
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);

        // Draw label and score
        std::string label_text = class_name + ": " + std::to_string(score).substr(0, 4);
        int baseline;
        cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        cv::rectangle(image,
          cv::Point(x1, y1 - text_size.height - baseline),
          cv::Point(x1 + text_size.width, y1),
          cv::Scalar(0, 0, 255), -1);

        cv::putText(image, label_text, cv::Point(x1, y1 - baseline),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
      }
    }
  }

private:
  torch::jit::script::Module model;
  torch::Device device;
};

int main(int argc, char* argv[])
{
  try {
    // Check arguments
    if (argc < 3) {
      std::cout << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
      return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // Initialize detector
    bool use_cuda = torch::cuda::is_available();
    FCOSDetector detector(model_path, use_cuda);

    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Error: Could not load image from " << image_path << std::endl;
      return -1;
    }

    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

    std::cout << "Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Run inference
    auto [boxes, scores, labels] = detector.predict(image_rgb);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time: " << duration.count() << "ms" << std::endl;

    // Draw predictions
    detector.draw_predictions(image, boxes, scores, labels, 0.5f);

    // Show result
    cv::imshow("FCOS Object Detection", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Optionally save result
    cv::imwrite("result.jpg", image);
    std::cout << "Result saved as 'result.jpg'" << std::endl;

  } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return -1;
  }

    return 0;
}
