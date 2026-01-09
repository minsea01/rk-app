#include <iostream>
#include <thread>
#include <vector>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "rkapp/infer/RknnEngine.hpp"

struct Item { int id; cv::Mat img; };

int main(int argc, char** argv){
  if(argc < 3){
    std::cerr << "Usage: " << argv[0] << " <rknn_model> <video or folder>" << std::endl;
    return 1;
  }
  const std::string model = argv[1];
  const std::string src = argv[2];

  // Producer: capture frames (from video/file pattern)
  cv::VideoCapture cap;
  if(std::filesystem::is_directory(src)){
    std::cerr << "Folder input not implemented in this sample" << std::endl; return 1;
  } else {
    cap.open(src);
  }
  if(!cap.isOpened()){ std::cerr << "Failed to open source: " << src << std::endl; return 1; }

  // Create 3 RKNN engines, bind to 3 cores (best-effort)
  constexpr int K = 3;
  std::vector<std::unique_ptr<rkapp::infer::RknnEngine>> engines;
  engines.reserve(K);
  for(int i=0;i<K;i++){
    auto eng = std::make_unique<rkapp::infer::RknnEngine>();
    eng->setCoreMask(1u << i);
    if(!eng->init(model, 640)){
      std::cerr << "Engine init failed for core " << i << std::endl; return 1;
    }
    engines.push_back(std::move(eng));
  }

  // Bounded queue
  std::mutex mtx; std::condition_variable cv_not_full, cv_not_empty; std::deque<Item> q; const size_t QMAX = 4;
  std::atomic<bool> done{false}; std::atomic<int> next_id{0};

  // Producer thread - 显式捕获避免悬垂引用风险
  std::thread t_cap([&cap, &mtx, &cv_not_full, &cv_not_empty, &q, QMAX, &next_id, &done]{
    cv::Mat f;
    while(cap.read(f)){
      std::unique_lock<std::mutex> lk(mtx);
      cv_not_full.wait(lk, [&q, QMAX]{ return q.size() < QMAX; });
      q.push_back(Item{next_id++, f.clone()});
      lk.unlock(); cv_not_empty.notify_one();
    }
    std::lock_guard<std::mutex> lk(mtx); done = true; cv_not_empty.notify_all();
  });

  // Worker threads mapped to engines - 显式捕获
  std::atomic<int> processed{0};
  auto worker = [&engines, &mtx, &cv_not_full, &cv_not_empty, &q, &done, &processed](int idx){
    auto& eng = *engines[idx];
    while(true){
      Item it{}; bool has=false;
      {
        std::unique_lock<std::mutex> lk(mtx);
        cv_not_empty.wait(lk, [&q, &done]{ return !q.empty() || done; });
        if(!q.empty()){ it = std::move(q.front()); q.pop_front(); has=true; cv_not_full.notify_one(); }
        else if(done) break;
      }
      if(!has) break;
      auto dets = eng.infer(it.img);
      (void)dets; // 演示：此处省略后处理
      processed++;
      if(it.id % 30 == 0){ std::cout << "core " << idx << " processed frame " << it.id << std::endl; }
    }
  };

  std::vector<std::thread> workers; workers.reserve(K);
  for(int i=0;i<K;i++) workers.emplace_back(worker, i);
  for(auto& th: workers) th.join();
  t_cap.join();

  std::cout << "Processed: " << processed.load() << " frames" << std::endl;
  return 0;
}


