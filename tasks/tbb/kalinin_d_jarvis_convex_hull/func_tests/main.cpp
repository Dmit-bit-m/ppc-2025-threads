// Copyright 2025 Dmitry Kalinin
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/kalinin_d_jarvis_convex_hull/include/ops_tbb.hpp"

TEST(kalinin_d_jarvis_convex_hull_tbb, Two_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> points = {{.x = 0, .y = 0}, {.x = 1, .y = 1}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> hull = {{.x = 0, .y = 0}, {.x = 1, .y = 1}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(res_hull[0], hull[0]);
  ASSERT_EQ(res_hull[1], hull[1]);
}

TEST(kalinin_d_jarvis_convex_hull_tbb, Duplicate_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> points = {{.x = 0, .y = 0}, {.x = 1, .y = 1}, {.x = 2, .y = 2},
                                                                 {.x = 0, .y = 0}, {.x = 1, .y = 1}, {.x = 2, .y = 2}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> hull = {{.x = 0, .y = 0}, {.x = 2, .y = 2}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());
  // Create Task
  kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(res_hull[0], hull[0]);
  ASSERT_EQ(res_hull[1], hull[1]);
}

TEST(kalinin_d_jarvis_convex_hull_tbb, Random_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> points = {{.x = 1, .y = 4}, {.x = 3, .y = 8}, {.x = 8, .y = 2},
                                                                 {.x = 5, .y = 5}, {.x = 9, .y = 1}, {.x = 4, .y = 7}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> hull = {
      {.x = 1, .y = 4}, {.x = 3, .y = 8}, {.x = 4, .y = 7}, {.x = 9, .y = 1}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> res_hull(hull.size());
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(res_hull[i], hull[i]);
  }
}

TEST(kalinin_d_jarvis_convex_hull_tbb, Rectangle_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> points = {{.x = 0, .y = 0}, {.x = 0, .y = 5}, {.x = 5, .y = 5},
                                                                 {.x = 5, .y = 0}, {.x = 1, .y = 1}, {.x = 1, .y = 4},
                                                                 {.x = 4, .y = 4}, {.x = 4, .y = 1}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> hull = {
      {.x = 0, .y = 0}, {.x = 0, .y = 5}, {.x = 5, .y = 5}, {.x = 5, .y = 0}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(res_hull[i], hull[i]);
  }
}

TEST(kalinin_d_jarvis_convex_hull_tbb, Star_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> points = {
      {.x = 0, .y = 3}, {.x = 1, .y = 1}, {.x = 2, .y = 3}, {.x = 3, .y = 1}, {.x = 4, .y = 3},
      {.x = 2, .y = 0}, {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 2, .y = 4}, {.x = 2, .y = -1}};
  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> hull = {{.x = 0, .y = 0}, {.x = 0, .y = 3}, {.x = 2, .y = 4},
                                                               {.x = 4, .y = 3}, {.x = 4, .y = 0}, {.x = 2, .y = -1}};

  std::vector<kalinin_d_jarvis_convex_hull_tbb::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_tbb::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(res_hull[i], hull[i]);
  }
}
