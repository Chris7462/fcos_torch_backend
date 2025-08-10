#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <unordered_map>
#include <string>

namespace config
{

// COCO class labels mapping using actual category IDs (with gaps)
// This maps from the actual COCO category ID to the class name
const std::unordered_map<int, std::string> COCO_CLASSES_NAME_MAP = {
  {0, "__background__"},  // Background class (though typically not returned in detections)
  {1, "person"}, {2, "bicycle"}, {3, "car"}, {4, "motorcycle"}, {5, "airplane"},
  {6, "bus"}, {7, "train"}, {8, "truck"}, {9, "boat"}, {10, "traffic light"},
  {11, "fire hydrant"}, {13, "stop sign"}, {14, "parking meter"}, {15, "bench"},
  {16, "bird"}, {17, "cat"}, {18, "dog"}, {19, "horse"}, {20, "sheep"}, {21, "cow"},
  {22, "elephant"}, {23, "bear"}, {24, "zebra"}, {25, "giraffe"}, {27, "backpack"},
  {28, "umbrella"}, {31, "handbag"}, {32, "tie"}, {33, "suitcase"}, {34, "frisbee"},
  {35, "skis"}, {36, "snowboard"}, {37, "sports ball"}, {38, "kite"},
  {39, "baseball bat"}, {40, "baseball glove"}, {41, "skateboard"}, {42, "surfboard"},
  {43, "tennis racket"}, {44, "bottle"}, {46, "wine glass"}, {47, "cup"}, {48, "fork"},
  {49, "knife"}, {50, "spoon"}, {51, "bowl"}, {52, "banana"}, {53, "apple"},
  {54, "sandwich"}, {55, "orange"}, {56, "broccoli"}, {57, "carrot"}, {58, "hot dog"},
  {59, "pizza"}, {60, "donut"}, {61, "cake"}, {62, "chair"}, {63, "couch"},
  {64, "potted plant"}, {65, "bed"}, {67, "dining table"}, {70, "toilet"}, {72, "tv"},
  {73, "laptop"}, {74, "mouse"}, {75, "remote"}, {76, "keyboard"}, {77, "cell phone"},
  {78, "microwave"}, {79, "oven"}, {80, "toaster"}, {81, "sink"}, {82, "refrigerator"},
  {84, "book"}, {85, "clock"}, {86, "vase"}, {87, "scissors"}, {88, "teddy bear"},
  {89, "hair drier"}, {90, "toothbrush"}
};
// Note: IDs 12, 26, 29, 30, 45, 66, 68, 69, 71, 83 are not used
// (they were categories in early drafts of COCO but removed).

// Helper function to get class name from category ID
inline std::string get_class_name(int category_id)
{
  auto it = COCO_CLASSES_NAME_MAP.find(category_id);
  return (it != COCO_CLASSES_NAME_MAP.end()) ? it->second : "unknown";
}

} // namespace config
