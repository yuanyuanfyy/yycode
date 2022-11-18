#include "MxBase/Log/Log.h"
#include "Alphapose.h"

namespace {
const uint32_t DEVICE_ID = 0;
const char RESULT_PATH[] = "../data/";
}  // namespace

int main(int argc, char *argv[]) {
    if (argc <= 3) {
        LogWarn << "Please input image path, such as './ [om_file_path] [img_path] [dataset_name]'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;


    initParam.checkTensor = true;

    initParam.modelPath = argv[1];
    auto inferAlphapose = std::make_shared<Alphapose>();
    APP_ERROR ret = inferAlphapose->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Alphapose init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[2];
    std::string dataset_name = argv[3];
    ret = inferAlphapose->Process(imgPath, RESULT_PATH, dataset_name);
    if (ret != APP_ERR_OK) {
        LogError << "Alphapose process failed, ret=" << ret << ".";
        inferAlphapose->DeInit();
        return ret;
    }
    inferAlphapose->DeInit();
    return APP_ERR_OK;
}
