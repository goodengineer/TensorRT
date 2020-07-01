/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


//Modifications by P.E.Mazniker(http://swhwengineering.webs.com

#ifndef MASKRCNN_CONFIG_HEADER
#define MASKRCNN_CONFIG_HEADER
#include "NvInfer.h"
#include <string>
#include <vector>
using namespace nvinfer1;

namespace MaskRCNNConfig
{
static const nvinfer1::Dims3 IMAGE_SHAPE{3, 1024, 1024};

// Pooled ROIs
static const int POOL_SIZE = 7;
static const int MASK_POOL_SIZE = 14;

// Threshold to determine the mask area out of final convolution output
static const float MASK_THRESHOLD = 0.5;

// Bounding box refinement standard deviation for RPN and final detections.
static const float RPN_BBOX_STD_DEV[] = {0.1, 0.1, 0.2, 0.2};
static const float BBOX_STD_DEV[] = {0.1, 0.1, 0.2, 0.2};

// Max number of final detections
static const int DETECTION_MAX_INSTANCES = 200;

// Minimum probability value to accept a detected instance
// ROIs below this threshold are skipped
static const float DETECTION_MIN_CONFIDENCE = 0.7;

// Non-maximum suppression threshold for detection
static const float DETECTION_NMS_THRESHOLD = 0.3;

// The strides of each layer of the FPN Pyramid. These values
// are based on a Resnet101 backbone.
static const std::vector<float> BACKBONE_STRIDES = {4, 8, 16, 32, 64};

// Size of the fully-connected layers in the classification graph
static const int FPN_CLASSIF_FC_LAYERS_SIZE = 1024;

// Size of the top-down layers used to build the feature pyramid
static const int TOP_DOWN_PYRAMID_SIZE = 256;

// Number of classification classes (including background)
static const int NUM_CLASSES = 1 + 72; // Custom Dental has 72 classes(asper VoTT project)

// Length of square anchor side in pixels
static const std::vector<float> RPN_ANCHOR_SCALES = {32, 64, 128, 256, 512};

// Ratios of anchors at each cell (width/height)
// A value of 1 represents a square anchor, and 0.5 is a wide anchor
static const float RPN_ANCHOR_RATIOS[] = {0.5, 1, 2};

// Anchor stride
// If 1 then anchors are created for each cell in the backbone feature map.
// If 2, then anchors are created for every other cell, and so on.
static const int RPN_ANCHOR_STRIDE = 1;

// Although Python impementation uses 6000,
//  TRT fails if this number larger than MAX_TOPK_K defined in engine/checkMacros.h
static const int MAX_PRE_NMS_RESULTS = 1024; // 3840;

// Non-max suppression threshold to filter RPN proposals.
// You can increase this during training to generate more propsals.
static const float RPN_NMS_THRESHOLD = 0.7;

// ROIs kept after non-maximum suppression (training and inference)
static const int POST_NMS_ROIS_INFERENCE = 1000;

// Custom Dental Categories to Detect as per VoTT Project Class names
static const std::vector<std::string> CLASS_NAMES = {
    "tags": [
        "DecayA1",
        "Restoration",
        "EndodonticA1",
        "ImplantA1",
        "CrownA1",
        "CariesA1",
        "OcclusialFilling",
        "MessialFilling",
        "Composite(Amalgam)Filling",
        "LingualFilling(Apron)",
        "DistalFilling",
        "ApicalLesion",
        "Pontic",
        "AmalgamFilling",
        "BiteSplint",
        "Abutment",
        "Bridge",
        "Clasp",
        "MissingTooth",
        "GenericFilling",
        "GenericCaries",
        "PathogenicTeethGapA1",
        "ChippedOrBrokenTooth",
        "MultipleSurfaceFilling",
        "LabialFilling",
        "InferiorAlveolarNerve(IAN)",
        "AnteriorMentalNerve(AMN)Loop",
        "Bifid(Bilateral)MandibularChannel",
        "MoxillaryFissure",
        "RootPossition",
        "Tooth18",
        "Tooth17",
        "Tooth16",
        "Tooth15",
        "Tooth14",
        "Tooth12",
        "Tooth13",
        "Tooth11",
        "Tooth21",
        "Tooth22",
        "Tooth23",
        "Tooth24",
        "Tooth25",
        "Tooth26",
        "Tooth27",
        "Tooth28",
        "Tooth48",
        "Tooth47",
        "Tooth46",
        "Tooth45",
        "Tooth44",
        "Tooth43",
        "Tooth42",
        "Tooth41",
        "Tooth31",
        "Tooth32",
        "Tooth33",
        "Tooth34",
        "Tooth35",
        "Tooth36",
        "Tooth37",
        "Tooth38",
        "PeriapicalPathosis",
        "Veneer",
        "OnlayFilling",
        "InlayFilling",
        "SupernumeraryTooth",
        "MandibularTraumaDevice",
        "TeethApex",
        "Periodontitis",
        "PlaceForAttachment",
        "PlaceForHook",
};

static const std::string MODEL_NAME = "mrcnn_nchwd.uff";
static const std::string MODEL_INPUT = "input_image";
static const Dims3 MODEL_INPUT_SHAPE = IMAGE_SHAPE;
static const std::vector<std::string> MODEL_OUTPUTS = {"mrcnn_detection", "mrcnn_mask/Sigmoid"};
static const Dims2 MODEL_DETECTION_SHAPE{DETECTION_MAX_INSTANCES, 6};
static const Dims4 MODEL_MASK_SHAPE{DETECTION_MAX_INSTANCES, NUM_CLASSES, 28, 28};
} // namespace MaskRCNNConfig
#endif
