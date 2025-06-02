#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstring>
#include <getopt.h>
#include <atomic>

using namespace cv;
using namespace std;
using namespace Ort;
using namespace boost::asio;
using namespace boost::beast;

// Global variables
Mat mask;
mutex mask_mutex;
atomic<float> steering(90.0f);
io_context io_ctx;
unique_ptr<websocket::stream<tcp::socket>> ws;
Session* ort_session = nullptr;
atomic<bool> running(true);

// Command-line arguments struct
struct Args {
    string model_file = "optimized.onnx";
    string ip_address = "192.168.0.10";
    int speed = 50;
};

// Linear interpolation
float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

// Safe division
float safe_div(float x, float y) {
    return y == 0 ? 0 : x / y;
}

// Image processing
Mat imageProcessing(const Mat& frame, int hl, int sl, int vl, int hu, int su, int vu) {
    Mat hsv, blur, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    GaussianBlur(hsv, blur, Size(5, 5), 0);
    inRange(blur, Scalar(hl, sl, vl), Scalar(hu, su, vu), mask);
    erode(mask, mask, getStructuringElement(MORPH_RECT, Size(5, 5)), Point(-1, -1), 1);
    dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(5, 5)), Point(-1, -1), 1);
    GaussianBlur(mask, blur, Size(5, 5), 0);
    return blur;
}

// Predict steering angle
void predictSteering(const string& model_file) {
    try {
        Env env(ORT_LOGGING_LEVEL_WARNING, "model");
        SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ort_session = new Session(env, model_file.c_str(), session_options);

        vector<int64_t> input_dims = {1, 66, 100, 1};
        vector<int64_t> output_dims = {1, 1};
        const char* input_names[] = {"input"};
        const char* output_names[] = {"dense_9"}; // Adjust based on your model's output layer name

        while (running) {
            Mat mask_copy;
            {
                lock_guard<mutex> lock(mask_mutex);
                if (mask.empty()) continue;
                mask_copy = mask.clone();
            }

            resize(mask_copy, mask_copy, Size(100, 66));
            mask_copy.convertTo(mask_copy, CV_32F);
            vector<float> input_data(mask_copy.begin<float>(), mask_copy.end<float>());
            auto input_tensor = Value::CreateTensor<float>(
                MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
                input_data.data(), input_data.size(), input_dims.data(), input_dims.size()
            );

            auto outputs = ort_session->Run(
                RunOptions{},
                input_names, &input_tensor, 1,
                output_names, output_dims.size()
            );

            float* output_data = outputs[0].GetTensorMutableData<float>();
            steering = *output_data;
            cout << "\rSent steering value: " << fixed << setprecision(2) << steering << "      " << flush;
        }
    } catch (const exception& e) {
        cerr << "Prediction Error: " << e.what() << endl;
    }
}

// Video capture
void captureVideo(const string& ip) {
    string pipeline = "http://192.168.0.10:8080/stream ! jpegdec ! videoconvert ! appsink";
    VideoCapture cap(pipeline, CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Error opening video stream. Check PiCar connection and IP: " << ip << endl;
        exit(1);
    }

    namedWindow("PiCar Video", WINDOW_NORMAL);
    namedWindow("PiCar Mask", WINDOW_NORMAL);
    float fps = 0, prev_time = chrono::duration<float>(chrono::system_clock::now().time_since_epoch()).count();

    while (running) {
        Mat frame;
        if (!cap.read(frame)) continue;

        float new_time = chrono::duration<float>(chrono::system_clock::now().time_since_epoch()).count();
        int hl = getTrackbarPos("Hue Lower", "Controls");
        int sl = getTrackbarPos("Sat Lower", "Controls");
        int vl = getTrackbarPos("Val Lower", "Controls");
        int hu = getTrackbarPos("Hue Upper", "Controls");
        int su = getTrackbarPos("Sat Upper", "Controls");
        int vu = getTrackbarPos("Val Upper", "Controls");

        Mat processed = imageProcessing(frame, hl, sl, vl, hu, su, vu);
        {
            lock_guard<mutex> lock(mask_mutex);
            mask = processed.clone();
        }

        putText(frame, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255));
        imshow("PiCar Video", frame);
        imshow("PiCar Mask", mask);

        // Send steering command
        try {
            string msg = "{\"steer\": " + to_string(steering.load()) + "}";
            ws->write(net::buffer(msg));
        } catch (const exception& e) {
            cerr << "Socket Error: " << e.what() << endl;
        }

        fps = lerp(fps, safe_div(1, new_time - prev_time), 0.001);
        prev_time = new_time;

        if (waitKey(1) == 27) {
            running = false;
            break;
        }
    }
    cap.release();
    destroyAllWindows();
}

// Update speed
void updateSpeed(int x, void*) {
    try {
        string msg = "{\"drive\": " + to_string(x) + "}";
        ws->write(net::buffer(msg));
    } catch (const exception& e) {
        cerr << "Socket Error: " << e.what() << endl;
    }
}

// Add trackbars
void addControls() {
    namedWindow("Controls", WINDOW_NORMAL);
    resizeWindow("Controls", 300, 300);
    createTrackbar("Hue Lower", "Controls", nullptr, 255, nullptr); setTrackbarPos("Hue Lower", "Controls", 40);
    createTrackbar("Sat Lower", "Controls", nullptr, 255, nullptr); setTrackbarPos("Sat Lower", "Controls", 25);
    createTrackbar("Val Lower", "Controls", nullptr, 255, nullptr); setTrackbarPos("Val Lower", "Controls", 73);
    createTrackbar("Hue Upper", "Controls", nullptr, 255, nullptr); setTrackbarPos("Hue Upper", "Controls", 93);
    createTrackbar("Sat Upper", "Controls", nullptr, 255, nullptr); setTrackbarPos("Sat Upper", "Controls", 194);
    createTrackbar("Val Upper", "Controls", nullptr, 255, nullptr); setTrackbarPos("Val Upper", "Controls", 245);
    createTrackbar("Speed", "Controls", nullptr, 100, updateSpeed); setTrackbarPos("Speed", "Controls", 0);
}

// Parse arguments
Args parseArgs(int argc, char* argv[]) {
    Args args;
    static struct option long_options[] = {
        {"neural_network_file", required_argument, nullptr, 'n'},
        {"ip_address", required_argument, nullptr, 'i'},
        {"speed", required_argument, nullptr, 's'},
        {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "n:i:s:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'n': args.model_file = optarg; break;
            case 'i': args.ip_address = optarg; break;
            case 's': args.speed = stoi(optarg); break;
            default: exit(1);
        }
    }
    return args;
}

// Print config
void printConfig(const Args& args) {
    cout << "--------------------------------- Config ---------------------------------\n"
         << "Neural network file name: " << args.model_file << "\n"
         << "Car ip address: " << args.ip_address << "\n"
         << "Car speed: " << args.speed << "\n"
         << "--------------------------------------------------------------------------\n";
}

// Print banner
void printBanner() {
    cout << R"(
       ____ ___ ____             ____                              
      |  _ \_ _/ ___|__ _ _ __  |  _ \ _   _ _ __  _ __   ___ _ __ 
      | |_) | | |   / _` | '__| | |_) | | | | '_ \| '_ \ / _ \ '__|
      |  __/| | |__| (_| | |    |  _ <| |_| | | | | | | |  __/ |   
      |_|  |___\____\__,_|_|    |_| \_\\__,_|_| |_|_| |_|\___|_|   
      ______________________________________________________________                                                         
)" << endl;
}

// Connect to PiCar
void tryConnect(const string& ip) {
    try {
        tcp::resolver resolver(io_ctx);
        auto endpoints = resolver.resolve(ip, "3000");
        ws = make_unique<websocket::stream<tcp::socket>>(io_ctx);
        ws->connect(endpoints);
        ws->handshake(ip, "/");
    } catch (const exception& e) {
        cerr << "Failed to connect to PiCar Socket Error: " << e.what() << endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    printBanner();
    Args args = parseArgs(argc, argv);
    printConfig(args);
    tryConnect(args.ip_address);
    addControls();
    thread prediction_thread(predictSteering, args.model_file);
    captureVideo(args.ip_address);
    running = false;
    prediction_thread.join();
    ws->close(websocket::close_code::normal);
    if (ort_session) delete ort_session;
    return 0;
}
