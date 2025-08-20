#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <string>

// OpenCV includes
#include <opencv2/core.hpp>

// Torch includes
#include <torch/script.h>
#include <torch/torch.h>


namespace fcos_torch_backend
{
class FCOSTorchBackend
{
public:
  FCOSTorchBackend(const std::string & model_path, torch::Device device = torch::kCPU);

  // Run inference
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predict(const cv::Mat & image);

  void draw_predictions(
    cv::Mat & image, const torch::Tensor & boxes, const torch::Tensor & scores,
    const torch::Tensor & labels, float confidence_threshold = 0.5f);

private:
  // Convert OpenCV Mat to PyTorch tensor
  torch::Tensor mat_to_tensor(const cv::Mat & image);

private:
  torch::jit::script::Module model_;
  torch::Device device_;
};

} // namespace fcos_torch_backend
