package engine

import (
	"testing"
	"time"
)

func TestApplyDecayRemovesExpiredAndWeakEdges(t *testing.T) {
	graph := NewGraph()

	nodes := []struct {
		id       string
		nodeType NodeType
	}{
		{"a", UserNode},
		{"b", UserNode},
		{"c", UserNode},
		{"d", UserNode},
		{"x", UserNode},
	}

	for _, n := range nodes {
		err := graph.AddNode(&Node{ID: n.id, Type: n.nodeType, Metadata: map[string]interface{}{}})
		if err != nil {
			t.Fatalf("failed to add node %s: %v", n.id, err)
		}
	}

	if err := graph.AddEdge(&Edge{From: "a", To: "b", Weight: 1.0, EdgeType: "friend"}); err != nil {
		t.Fatalf("failed to add edge a->b: %v", err)
	}
	if err := graph.AddEdge(&Edge{From: "a", To: "c", Weight: 0.2, EdgeType: "friend"}); err != nil {
		t.Fatalf("failed to add edge a->c: %v", err)
	}
	if err := graph.AddEdge(&Edge{From: "a", To: "d", Weight: 0.9, EdgeType: "friend"}); err != nil {
		t.Fatalf("failed to add edge a->d: %v", err)
	}
	if err := graph.AddEdge(&Edge{From: "x", To: "b", Weight: 0.4, EdgeType: "friend"}); err != nil {
		t.Fatalf("failed to add edge x->b: %v", err)
	}

	now := time.Now()
	graph.edges["a"]["b"].UpdatedAt = now.Add(-200 * time.Hour) // beyond max age
	graph.edges["a"]["c"].UpdatedAt = now.Add(-48 * time.Hour)  // will decay below threshold
	graph.edges["a"]["d"].UpdatedAt = now.Add(-2 * time.Hour)
	graph.edges["x"]["b"].UpdatedAt = now.Add(-300 * time.Hour)

	graph.ApplyDecay(0.1, 168*time.Hour)

	if _, exists := graph.edges["a"]["b"]; exists {
		t.Fatal("expected a->b edge to be removed after exceeding max age")
	}
	if _, exists := graph.edges["a"]["c"]; exists {
		t.Fatal("expected a->c edge to be removed after decaying below threshold")
	}

	edgeAD, exists := graph.edges["a"]["d"]
	if !exists {
		t.Fatal("expected a->d edge to persist")
	}
	if edgeAD.Weight <= 0.1 || edgeAD.Weight >= 0.9 {
		t.Fatalf("expected a->d weight to be decayed but above threshold, got %f", edgeAD.Weight)
	}

	if _, exists := graph.edges["x"]; exists {
		t.Fatal("expected x edge map to be removed when all edges decayed")
	}
}
