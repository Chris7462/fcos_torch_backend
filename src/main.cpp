#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include "fcos_torch_backend/config.hpp"
#include "fcos_torch_backend/fcos_torch_backend.hpp"


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
    fcos_torch_backend::FCOSTorchBackend detector(model_path);

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
    detector.draw_predictions(image, boxes, scores, labels, 0.6f);

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
