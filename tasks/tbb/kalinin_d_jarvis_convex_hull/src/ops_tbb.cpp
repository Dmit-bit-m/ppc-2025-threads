// Copyright 2025 Kalinin Dmitry
#include "tbb/kalinin_d_jarvis_convex_hull/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <vector>

using namespace std::chrono_literals;

namespace kalinin_d_jarvis_convex_hull_tbb {

Point FindStartingPoint(const std::vector<Point>& points) {
  Point p0 = points[0];
  for (const auto& p : points) {
    if (p.x < p0.x || (p.x == p0.x && p.y < p0.y)) {
      p0 = p;
    }
  }
  return p0;
}

Point FindNextPoint(const Point& prev_point, const std::vector<Point>& points) {
  Point next_point = points[0];
  std::mutex mutex;

  tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()), [&](const tbb::blocked_range<size_t>& range) {
    Point local_next_point = next_point;

    for (size_t i = range.begin(); i < range.end(); ++i) {
      const auto& point = points[i];
      if (point == prev_point) {
        continue;
      }

      double cross_product = ((point.y - prev_point.y) * (local_next_point.x - prev_point.x)) -
                             ((point.x - prev_point.x) * (local_next_point.y - prev_point.y));

      if (cross_product > 0 ||
          (cross_product == 0 &&
           ((point.x - prev_point.x) * (point.x - prev_point.x) + (point.y - prev_point.y) * (point.y - prev_point.y)) >
               ((local_next_point.x - prev_point.x) * (local_next_point.x - prev_point.x) +
                (local_next_point.y - prev_point.y) * (local_next_point.y - prev_point.y)))) {
        local_next_point = point;
      }
    }

    // Потокобезопасное обновление глобального next_point
    std::lock_guard<std::mutex> lock(mutex);
    double cross_product = ((local_next_point.y - prev_point.y) * (next_point.x - prev_point.x)) -
                           ((local_next_point.x - prev_point.x) * (next_point.y - prev_point.y));

    if (cross_product > 0 ||
        (cross_product == 0 && ((local_next_point.x - prev_point.x) * (local_next_point.x - prev_point.x) +
                                (local_next_point.y - prev_point.y) * (local_next_point.y - prev_point.y)) >
                                   ((next_point.x - prev_point.x) * (next_point.x - prev_point.x) +
                                    (next_point.y - prev_point.y) * (next_point.y - prev_point.y)))) {
      next_point = local_next_point;
    }
  });

  return next_point;
}

std::vector<Point> Jarvis(const std::vector<Point>& points) {
  if (points.size() < 3) {
    return points;
  }

  Point p0 = FindStartingPoint(points);
  std::vector<Point> convex_hull = {p0};
  Point prev_point = p0;

  while (true) {
    Point next_point = FindNextPoint(prev_point, points);
    if (next_point == p0) {
      break;
    }
    convex_hull.push_back(next_point);
    prev_point = next_point;
  }

  return convex_hull;
}

}  // namespace kalinin_d_jarvis_convex_hull_tbb

bool kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential::PreProcessingImpl() {
  points_.resize(task_data->inputs_count[0]);
  auto* tmp_ptr_a = reinterpret_cast<kalinin_d_jarvis_convex_hull_tbb::Point*>(task_data->inputs[0]);
  std::copy_n(tmp_ptr_a, task_data->inputs_count[0], points_.begin());
  return true;
}

bool kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs_count[0] == 0) {
    return false;
  }

  std::ranges::sort(points_, [](const Point& a, const Point& b) { return a < b; });
  return std::ranges::unique(points_).begin() == points_.end();
}

bool kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential::RunImpl() {
  convexHullPoints_ = kalinin_d_jarvis_convex_hull_tbb::Jarvis(points_);
  return true;
}

bool kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<kalinin_d_jarvis_convex_hull_tbb::Point*>(task_data->outputs[0]);
  std::copy_n(convexHullPoints_.begin(), convexHullPoints_.size(), output_ptr);
  return true;
}
