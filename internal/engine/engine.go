package engine

import (
	"fmt"
	"time"
)

// Engine wraps the GNN and Graph for API usage
type Engine struct {
	graph *Graph
	gnn   *GNN
}

// NewEngine creates a new SocialGNN engine
func NewEngine() *Engine {
	graph := NewGraph()
	gnn := NewGNN(graph, 128, MeanAggregation)

	return &Engine{
		graph: graph,
		gnn:   gnn,
	}
}

// AddNode adds a node to the graph
func (e *Engine) AddNode(id string, nodeType NodeType, metadata map[string]interface{}) error {
	node := &Node{
		ID:       id,
		Type:     nodeType,
		Metadata: metadata,
	}

	if err := e.graph.AddNode(node); err != nil {
		return fmt.Errorf("failed to add node: %w", err)
	}

	// Initialize embedding for new node (only for this node to avoid O(n²) behavior)
	e.gnn.InitializeNodeEmbedding(node.ID)

	return nil
}

// addNodeWithoutEmbeddingInit adds a node without initializing embeddings (for batch operations)
func (e *Engine) addNodeWithoutEmbeddingInit(id string, nodeType NodeType, metadata map[string]interface{}) error {
	node := &Node{
		ID:       id,
		Type:     nodeType,
		Metadata: metadata,
	}

	if err := e.graph.AddNode(node); err != nil {
		return fmt.Errorf("failed to add node: %w", err)
	}

	return nil
}

// AddEdge adds an edge between nodes
func (e *Engine) AddEdge(from, to string, weight float64, edgeType string) error {
	edge := &Edge{
		From:     from,
		To:       to,
		Weight:   weight,
		EdgeType: edgeType,
	}

	if err := e.graph.AddEdge(edge); err != nil {
		return fmt.Errorf("failed to add edge: %w", err)
	}

	return nil
}

// GetRecommendations returns recommendations for a user
func (e *Engine) GetRecommendations(userID string, nodeType NodeType, limit int) ([]*Node, error) {
	recommendations, err := e.gnn.GetRecommendations(userID, nodeType, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get recommendations: %w", err)
	}

	return recommendations, nil
}

// GetSimilarity calculates similarity between two users
func (e *Engine) GetSimilarity(user1ID, user2ID string) (float64, error) {
	embedding1, err := e.gnn.MessagePassing(user1ID, 2)
	if err != nil {
		return 0, fmt.Errorf("failed to get embedding for user1: %w", err)
	}

	embedding2, err := e.gnn.MessagePassing(user2ID, 2)
	if err != nil {
		return 0, fmt.Errorf("failed to get embedding for user2: %w", err)
	}

	similarity := e.gnn.CosineSimilarity(embedding1, embedding2)
	return similarity, nil
}

// UpdateEngagement updates engagement metrics for a node and affects rankings
func (e *Engine) UpdateEngagement(nodeID string, engagementType string, delta int) error {
	return e.graph.UpdateEngagement(nodeID, engagementType, delta)
}

// GetEngagementScore returns the engagement score for a node
func (e *Engine) GetEngagementScore(nodeID string) (float64, error) {
	return e.graph.GetEngagementScore(nodeID)
}

// GetPersuasivenessScore returns the persuasiveness score for a user
func (e *Engine) GetPersuasivenessScore(userID string) (float64, error) {
	return e.graph.GetPersuasivenessScore(userID)
}

// GetSpamScore returns the spam score for a post
func (e *Engine) GetSpamScore(nodeID string) (float64, error) {
	return e.graph.GetSpamScore(nodeID)
}

// GetRankedComments returns intelligently ranked comments for a post
func (e *Engine) GetRankedComments(postID string) ([]*CommentRankingResult, error) {
	return e.graph.GetRankedComments(postID)
}

// GetStats returns graph statistics
func (e *Engine) GetStats() map[string]interface{} {
	return e.graph.Stats()
}

// GetNeighbors returns all neighbors of a node
func (e *Engine) GetNeighbors(nodeID string) (map[string]*Edge, bool) {
	return e.graph.GetNeighbors(nodeID)
}

// GetUserSpamFlags returns spam detection flags for a user
func (e *Engine) GetUserSpamFlags(userID string) (*UserSpamMetrics, error) {
	return e.graph.GetUserSpamFlags(userID)
}

// GetFollowbackMetrics returns followback metrics for a user
func (e *Engine) GetFollowbackMetrics(userID string) (*FollowbackMetrics, error) {
	metrics := e.graph.calculateFollowbackMetrics(userID)
	if metrics.Following == 0 && metrics.Followers == 0 && metrics.Followbacks == 0 {
		return nil, fmt.Errorf("user %s not found or has no follow relationships", userID)
	}
	return metrics, nil
}

// UpdateFollowbackMetrics refreshes followback metrics for a user
func (e *Engine) UpdateFollowbackMetrics(userID string) {
	e.graph.updateFollowbackMetrics(userID)
}

// ApplyDecay applies temporal decay to edges
func (e *Engine) ApplyDecay() {
	e.graph.ApplyDecay(0.1, 168*time.Hour) // 1 week max age
}

// LoadSampleData loads the Riverside community sample data
func (e *Engine) LoadSampleData() {
	// Add sample users
	users := []map[string]interface{}{
		{"id": "sarah", "name": "Sarah Mitchell", "job": "Teacher", "interests": []string{"education", "hiking", "photography"}},
		{"id": "jake", "name": "Jake Rodriguez", "job": "Mechanic", "interests": []string{"cars", "fishing", "sports"}},
		{"id": "emma", "name": "Emma Chen", "job": "Software Developer", "interests": []string{"technology", "cooking", "travel"}},
		{"id": "tom", "name": "Tom Weatherby", "job": "Retired Rancher", "interests": []string{"woodworking", "history", "gardening"}},
	}

	for _, user := range users {
		err := e.addNodeWithoutEmbeddingInit(user["id"].(string), UserNode, user)
		if err != nil {
			fmt.Printf("Error adding user %s: %v\n", user["id"], err)
			return
		}
	}

	// Add sample posts
	posts := []map[string]interface{}{
		{"id": "post1", "content": "Beautiful sunrise over the Wind River Mountains! 🌄", "author": "sarah", "category": "nature"},
		{"id": "post2", "content": "Road construction on Highway 191 - expect delays", "author": "jake", "category": "local"},
		{"id": "post3", "content": "Fresh vegetables at Saturday farmers market!", "author": "emma", "category": "business"},
	}

	for _, post := range posts {
		err := e.addNodeWithoutEmbeddingInit(post["id"].(string), PostNode, post)
		if err != nil {
			fmt.Printf("Error adding post %s: %v\n", post["id"], err)
			return
		}
	}

	// Add sample connections
	e.AddEdge("sarah", "jake", 0.8, "friend")
	e.AddEdge("jake", "sarah", 0.8, "friend")
	e.AddEdge("sarah", "post1", 1.0, "author")
	e.AddEdge("jake", "post1", 0.7, "like")
	e.AddEdge("emma", "post1", 0.6, "like")

	// Initialize all embeddings once at the end for efficiency
	e.gnn.InitializeEmbeddings()
}

// Training methods for hybrid GNN
func (e *Engine) AddTrainingExample(userID, itemID string, rating float64) error {
	e.gnn.AddTrainingExample(userID, itemID, "interaction", rating)
	return nil
}

func (e *Engine) TrainBatch(epochs int) (float64, error) {
	err := e.gnn.TrainOnBatch(epochs)
	if err != nil {
		return 0.0, err
	}
	// Return a synthetic loss for now - could be improved with actual loss calculation
	return 0.1, nil
}

func (e *Engine) GetTrainingExampleCount() int {
	return len(e.gnn.TrainingData)
}