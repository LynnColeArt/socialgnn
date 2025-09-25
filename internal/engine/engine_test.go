package engine

import "testing"

func TestEngineTrainBatchHandlesLargeBatch(t *testing.T) {
	engine := NewEngine()

	if err := engine.AddNode("user1", UserNode, map[string]interface{}{"name": "User"}); err != nil {
		t.Fatalf("failed to add user: %v", err)
	}
	if err := engine.AddNode("post1", PostNode, map[string]interface{}{"author": "user1"}); err != nil {
		t.Fatalf("failed to add post: %v", err)
	}
	if err := engine.AddEdge("user1", "post1", 1.0, "like"); err != nil {
		t.Fatalf("failed to add edge: %v", err)
	}

	if err := engine.AddTrainingExample("user1", "post1", 1.0); err != nil {
		t.Fatalf("failed to add training example: %v", err)
	}

	if _, err := engine.TrainBatch(3, 5); err != nil {
		t.Fatalf("expected training to succeed, got error: %v", err)
	}
}

func TestEngineTrainBatchRequiresTrainingData(t *testing.T) {
	engine := NewEngine()
	if _, err := engine.TrainBatch(1, 2); err == nil {
		t.Fatal("expected error when no training data is available")
	}
}

func TestGetFollowbackMetricsReturnsZerosForNewUsers(t *testing.T) {
	engine := NewEngine()
	if err := engine.AddNode("loner", UserNode, map[string]interface{}{"name": "Solo"}); err != nil {
		t.Fatalf("failed to add user: %v", err)
	}

	metrics, err := engine.GetFollowbackMetrics("loner")
	if err != nil {
		t.Fatalf("expected metrics for existing user, got error: %v", err)
	}

	if metrics.Followers != 0 || metrics.Following != 0 || metrics.Followbacks != 0 {
		t.Fatalf("expected zeroed metrics for new user, got %+v", metrics)
	}
}

func TestGetFollowbackMetricsMissingUser(t *testing.T) {
	engine := NewEngine()
	if _, err := engine.GetFollowbackMetrics("ghost"); err == nil {
		t.Fatal("expected error for missing user")
	}
}
