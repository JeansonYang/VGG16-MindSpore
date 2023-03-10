#include <experimental/filesystem>
#include "Vgg16Classify.h"
#include "MxBase/Log/Log.h"

namespace fs = std::experimental::filesystem;
namespace {
const uint32_t CLASS_NUM = 1000;
}
std::vector<double> g_inferCost;

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './main ../data/input 10'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../data/config/imagenet1000_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/vgg16.om";
    auto vgg16 = std::make_shared<Vgg16Classify>();
    APP_ERROR ret = vgg16->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Vgg16Classify init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgDir = argv[1];
    int limit = std::strtol(argv[2], nullptr, 0);
    int index = 0;
    for (auto & entry : fs::directory_iterator(imgDir)) {
        if (index == limit) {
            break;
        }
        index++;
        LogInfo << "read image path " << entry.path();
        ret = vgg16->Process(entry.path());
        if (ret != APP_ERR_OK) {
            LogError << "Vgg16Classify process failed, ret=" << ret << ".";
            vgg16->DeInit();
            return ret;
        }
    }
    vgg16->DeInit();
    double costSum = 0;
    for (unsigned int i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}
