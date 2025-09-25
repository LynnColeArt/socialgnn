package engine

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// AggregationType defines how to aggregate neighbor embeddings
type AggregationType int

const (
	MeanAggregation AggregationType = iota
	MaxAggregation
	SumAggregation
	AttentionAggregation
)

// GNNLayer represents a single graph neural network layer
type GNNLayer struct {
	WeightMatrix [][]float64 // Input to hidden weights
	BiasVector   []float64   // Bias terms
	InputDim     int
	OutputDim    int
}

// HybridGNN combines learned embeddings with heuristic features
type HybridGNN struct {
	Graph           *Graph
	EmbeddingDim    int
	HeuristicDim    int         // Dimension of hand-crafted features
	Layers          []*GNNLayer // Neural network layers
	AggregationType AggregationType
	LearningRate    float64
	TrainingData    []*TrainingExample
	mu              sync.RWMutex
}

// TrainingExample represents a training sample for the GNN
type TrainingExample struct {
	UserID      string  `json:"user_id"`
	TargetID    string  `json:"target_id"`
	Interaction string  `json:"interaction"` // "like", "follow", "share", etc.
	Label       float64 `json:"label"`       // 1.0 for positive, 0.0 for negative
	Timestamp   int64   `json:"timestamp"`
}

// Legacy GNN for backward compatibility
type GNN struct {
	*HybridGNN
}

// NewHybridGNN creates a new hybrid GNN with proper neural layers
func NewHybridGNN(graph *Graph, embeddingDim, heuristicDim int, aggType AggregationType) *HybridGNN {
	// Create 2-layer neural network: input -> hidden -> output
	hiddenDim := 64
	layers := []*GNNLayer{
		// Layer 1: (embedding + heuristic) -> hidden
		{
			InputDim:  embeddingDim + heuristicDim,
			OutputDim: hiddenDim,
		},
		// Layer 2: hidden -> final embedding
		{
			InputDim:  hiddenDim,
			OutputDim: embeddingDim,
		},
	}

	// Initialize weights with Xavier initialization
	for _, layer := range layers {
		layer.WeightMatrix = make([][]float64, layer.InputDim)
		for i := range layer.WeightMatrix {
			layer.WeightMatrix[i] = make([]float64, layer.OutputDim)
			for j := range layer.WeightMatrix[i] {
				// Xavier initialization
				bound := math.Sqrt(6.0 / float64(layer.InputDim+layer.OutputDim))
				layer.WeightMatrix[i][j] = (rand.Float64() - 0.5) * 2.0 * bound
			}
		}
		layer.BiasVector = make([]float64, layer.OutputDim)
	}

	return &HybridGNN{
		Graph:           graph,
		EmbeddingDim:    embeddingDim,
		HeuristicDim:    heuristicDim,
		Layers:          layers,
		AggregationType: aggType,
		LearningRate:    0.01,
		TrainingData:    make([]*TrainingExample, 0),
	}
}

// NewGNN creates a new GNN instance (legacy compatibility)
func NewGNN(graph *Graph, embeddingDim int, aggType AggregationType) *GNN {
	heuristicDim := 8 // Persuasiveness, engagement, spam, followback, etc.
	hybridGNN := NewHybridGNN(graph, embeddingDim, heuristicDim, aggType)
	return &GNN{HybridGNN: hybridGNN}
}

// InitializeEmbeddings initializes random embeddings for all nodes
func (gnn *GNN) InitializeEmbeddings() {
	// Get all node types and process them
	nodeTypes := []NodeType{UserNode, PostNode, BusinessNode, EventNode, CommentNode}

	for _, nodeType := range nodeTypes {
		nodes := gnn.Graph.GetNodesByType(nodeType)
		for _, node := range nodes {
			if len(node.Embedding) == 0 {
				node.Embedding = make([]float64, gnn.EmbeddingDim)
				for i := range node.Embedding {
					// Xavier initialization
					node.Embedding[i] = (rand.Float64() - 0.5) * 2.0 * math.Sqrt(6.0/float64(gnn.EmbeddingDim))
				}
			}
		}
	}
}

// InitializeNodeEmbedding initializes embedding for a single node (more efficient than InitializeEmbeddings for single nodes)
func (gnn *GNN) InitializeNodeEmbedding(nodeID string) {
	node, exists := gnn.Graph.GetNode(nodeID)
	if !exists {
		return
	}

	if len(node.Embedding) == 0 {
		node.Embedding = make([]float64, gnn.EmbeddingDim)
		for i := range node.Embedding {
			// Xavier initialization
			node.Embedding[i] = (rand.Float64() - 0.5) * 2.0 * math.Sqrt(6.0/float64(gnn.EmbeddingDim))
		}
	}
}

// MessagePassing performs hybrid message passing (backward compatible)
func (gnn *GNN) MessagePassing(nodeID string, maxHops int) ([]float64, error) {
	// Use hybrid approach if available, fallback to legacy
	if gnn.HybridGNN != nil {
		return gnn.HybridGNN.HybridMessagePassing(nodeID, maxHops)
	}

	// Legacy fallback (shouldn't happen with new constructor)
	return gnn.legacyMessagePassing(nodeID, maxHops)
}

// legacyMessagePassing performs the old message passing for compatibility
func (gnn *GNN) legacyMessagePassing(nodeID string, maxHops int) ([]float64, error) {
	gnn.mu.RLock()
	defer gnn.mu.RUnlock()

	node, exists := gnn.Graph.GetNode(nodeID)
	if !exists {
		return nil, fmt.Errorf("node %s not found", nodeID)
	}

	// Start with the node's own embedding
	result := make([]float64, gnn.EmbeddingDim)
	copy(result, node.Embedding)

	// Collect neighbor embeddings
	neighbors, _ := gnn.Graph.GetNeighbors(nodeID)
	if len(neighbors) == 0 {
		return result, nil
	}

	neighborEmbeddings := make([][]float64, 0, len(neighbors))
	weights := make([]float64, 0, len(neighbors))

	for neighborID, edge := range neighbors {
		if neighborNode, exists := gnn.Graph.GetNode(neighborID); exists {
			neighborEmbeddings = append(neighborEmbeddings, neighborNode.Embedding)
			weights = append(weights, edge.Weight)
		}
	}

	// Aggregate neighbor embeddings
	aggregated := gnn.aggregate(neighborEmbeddings, weights)

	// Combine own embedding with aggregated neighbors
	for i := range result {
		result[i] = 0.5*result[i] + 0.5*aggregated[i]
	}

	return result, nil
}

// aggregate combines neighbor embeddings based on aggregation type
func (gnn *GNN) aggregate(embeddings [][]float64, weights []float64) []float64 {
	if len(embeddings) == 0 {
		return make([]float64, gnn.EmbeddingDim)
	}

	result := make([]float64, gnn.EmbeddingDim)

	switch gnn.AggregationType {
	case MeanAggregation:
		totalWeight := 0.0
		for i, embedding := range embeddings {
			weight := weights[i]
			totalWeight += weight
			for j, val := range embedding {
				result[j] += val * weight
			}
		}
		if totalWeight > 0 {
			for j := range result {
				result[j] /= totalWeight
			}
		}

	case MaxAggregation:
		for j := 0; j < gnn.EmbeddingDim; j++ {
			maxVal := math.Inf(-1)
			for i, embedding := range embeddings {
				val := embedding[j] * weights[i]
				if val > maxVal {
					maxVal = val
				}
			}
			result[j] = maxVal
		}

	case SumAggregation:
		for i, embedding := range embeddings {
			weight := weights[i]
			for j, val := range embedding {
				result[j] += val * weight
			}
		}

	case AttentionAggregation:
		// Simplified attention mechanism
		attentionWeights := gnn.computeAttention(embeddings, weights)
		for i, embedding := range embeddings {
			for j, val := range embedding {
				result[j] += val * attentionWeights[i]
			}
		}
	}

	return result
}

// computeAttention calculates attention weights for neighbors
func (gnn *GNN) computeAttention(embeddings [][]float64, weights []float64) []float64 {
	if len(embeddings) == 0 {
		return []float64{}
	}

	// Simplified attention: just normalize the edge weights with softmax
	attentions := make([]float64, len(embeddings))
	maxWeight := math.Inf(-1)

	for _, w := range weights {
		if w > maxWeight {
			maxWeight = w
		}
	}

	sum := 0.0
	for i, w := range weights {
		attentions[i] = math.Exp(w - maxWeight)
		sum += attentions[i]
	}

	if sum > 0 {
		for i := range attentions {
			attentions[i] /= sum
		}
	}

	return attentions
}

// extractHeuristicFeatures extracts hand-crafted features for a node
func (h *HybridGNN) extractHeuristicFeatures(nodeID string) []float64 {
	node, exists := h.Graph.GetNode(nodeID)
	if !exists || node.Engagement == nil {
		return make([]float64, h.HeuristicDim)
	}

	features := make([]float64, h.HeuristicDim)
	idx := 0

	// Persuasiveness (normalized)
	features[idx] = math.Min(1.0, node.Engagement.Persuasiveness/10.0)
	idx++

	// Engagement score (normalized)
	features[idx] = math.Min(1.0, node.Engagement.Score/20.0)
	idx++

	// Spam probability
	if node.Engagement.UserSpamFlags != nil {
		features[idx] = node.Engagement.UserSpamFlags.SpamProbability
	}
	idx++

	// Followback health score (for users)
	if node.Type == UserNode && node.Engagement.FollowbackMetrics != nil {
		features[idx] = node.Engagement.FollowbackMetrics.HealthScore / 1.5 // Normalize to 0-1
	} else {
		features[idx] = 1.0 // Default for non-users
	}
	idx++

	// Follow harvesting penalty
	if node.Engagement.UserSpamFlags != nil {
		features[idx] = node.Engagement.UserSpamFlags.FollowHarvesting
	}
	idx++

	// Node type encoding (one-hot-ish)
	switch node.Type {
	case UserNode:
		features[idx] = 1.0
	case PostNode:
		features[idx] = 0.5
	case CommentNode:
		features[idx] = 0.25
	default:
		features[idx] = 0.0
	}
	idx++

	// Engagement recency (time-based feature)
	now := time.Now()
	timeSinceUpdate := now.Sub(node.UpdatedAt).Hours()
	features[idx] = math.Max(0.0, 1.0-timeSinceUpdate/168.0) // Decay over 1 week
	idx++

	// Graph connectivity (degree centrality)
	neighbors, _ := h.Graph.GetNeighbors(nodeID)
	features[idx] = math.Min(1.0, float64(len(neighbors))/50.0) // Normalize by max expected degree

	return features
}

// forwardPass performs a forward pass through the neural network layers
func (h *HybridGNN) forwardPass(input []float64) []float64 {
	current := input
	for _, layer := range h.Layers {
		current = h.layerForward(current, layer)
		// Apply ReLU activation (except for last layer)
		if layer != h.Layers[len(h.Layers)-1] {
			for i := range current {
				current[i] = math.Max(0.0, current[i])
			}
		}
	}
	return current
}

// layerForward performs forward pass for a single layer: output = ReLU(input * W + b)
func (h *HybridGNN) layerForward(input []float64, layer *GNNLayer) []float64 {
	output := make([]float64, layer.OutputDim)

	// Matrix multiplication: input * WeightMatrix + bias
	for j := 0; j < layer.OutputDim; j++ {
		sum := layer.BiasVector[j]
		for i := 0; i < layer.InputDim && i < len(input); i++ {
			sum += input[i] * layer.WeightMatrix[i][j]
		}
		output[j] = sum
	}

	return output
}

// HybridMessagePassing combines learned embeddings with heuristic features (thread-safe)
func (h *HybridGNN) HybridMessagePassing(nodeID string, maxHops int) ([]float64, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.hybridMessagePassingCore(nodeID, maxHops)
}

// hybridMessagePassingCore is the core implementation without locking
// Used internally by training and recursive calls to avoid deadlock
func (h *HybridGNN) hybridMessagePassingCore(nodeID string, maxHops int) ([]float64, error) {
	node, exists := h.Graph.GetNode(nodeID)
	if !exists {
		return nil, fmt.Errorf("node %s not found", nodeID)
	}

	// 1. Get base embedding (learned or initialized)
	baseEmbedding := make([]float64, h.EmbeddingDim)
	if len(node.Embedding) == h.EmbeddingDim {
		copy(baseEmbedding, node.Embedding)
	} else {
		// Initialize with Xavier if not present
		for i := range baseEmbedding {
			baseEmbedding[i] = (rand.Float64() - 0.5) * 2.0 * math.Sqrt(6.0/float64(h.EmbeddingDim))
		}
	}

	// 2. Extract heuristic features
	heuristicFeatures := h.extractHeuristicFeatures(nodeID)

	// 3. Combine embedding + heuristic features
	combinedInput := make([]float64, h.EmbeddingDim+h.HeuristicDim)
	copy(combinedInput, baseEmbedding)
	copy(combinedInput[h.EmbeddingDim:], heuristicFeatures)

	// 4. Pass through neural network
	learnedEmbedding := h.forwardPass(combinedInput)

	// 5. Aggregate neighbor information (traditional message passing)
	neighbors, _ := h.Graph.GetNeighbors(nodeID)
	if len(neighbors) == 0 {
		return learnedEmbedding, nil
	}

	neighborEmbeddings := make([][]float64, 0, len(neighbors))
	weights := make([]float64, 0, len(neighbors))

	for neighborID, edge := range neighbors {
		if _, exists := h.Graph.GetNode(neighborID); exists {
			// Recursively get neighbor embedding (with hop limit)
			var neighborEmbedding []float64
			if maxHops > 0 {
				neighborEmbedding, _ = h.hybridMessagePassingCore(neighborID, maxHops-1)
			} else {
				neighborEmbedding = h.extractHeuristicFeatures(neighborID)
			}

			if len(neighborEmbedding) == h.EmbeddingDim {
				neighborEmbeddings = append(neighborEmbeddings, neighborEmbedding)
				weights = append(weights, edge.Weight)
			}
		}
	}

	// 6. Aggregate and combine with learned embedding
	if len(neighborEmbeddings) > 0 {
		aggregated := h.aggregateEmbeddings(neighborEmbeddings, weights)

		// Weighted combination: 70% learned embedding + 30% neighbor aggregation
		for i := range learnedEmbedding {
			if i < len(aggregated) {
				learnedEmbedding[i] = 0.7*learnedEmbedding[i] + 0.3*aggregated[i]
			}
		}
	}

	return learnedEmbedding, nil
}

// AddTrainingExample adds a training sample for the GNN to learn from
func (h *HybridGNN) AddTrainingExample(userID, targetID, interaction string, label float64) {
	h.mu.Lock()
	defer h.mu.Unlock()

	example := &TrainingExample{
		UserID:      userID,
		TargetID:    targetID,
		Interaction: interaction,
		Label:       label,
		Timestamp:   time.Now().Unix(),
	}

	h.TrainingData = append(h.TrainingData, example)

	// Keep only recent training data (last 10,000 examples)
	if len(h.TrainingData) > 10000 {
		h.TrainingData = h.TrainingData[len(h.TrainingData)-10000:]
	}
}

// TrainOnBatch performs a mini-batch training step using recent interactions
func (h *HybridGNN) TrainOnBatch(batchSize int) error {
	if batchSize <= 0 {
		batchSize = 1
	}

	// First, sample the batch with lock
	h.mu.Lock()
	if len(h.TrainingData) == 0 {
		h.mu.Unlock()
		return fmt.Errorf("no training data available")
	}

	// Sample random batch from recent training data
	batch := make([]*TrainingExample, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := rand.Intn(len(h.TrainingData))
		batch[i] = h.TrainingData[idx]
	}
	h.mu.Unlock() // Release lock before doing message passing

	// Compute gradients without holding lock
	gradientUpdates := make(map[string]float64)

	for _, example := range batch {
		// Get embeddings for user and target (uses unsafe version internally now)
		userEmbedding, err1 := h.hybridMessagePassingCore(example.UserID, 1)
		targetEmbedding, err2 := h.hybridMessagePassingCore(example.TargetID, 1)

		if err1 != nil || err2 != nil {
			continue // Skip invalid examples
		}

		// Compute prediction (cosine similarity)
		predicted := h.cosineSimilarity(userEmbedding, targetEmbedding)

		// Compute loss (mean squared error) - used for monitoring
		_ = (predicted - example.Label) * (predicted - example.Label)

		// Simple gradient update (this is a simplified version)
		gradient := 2.0 * (predicted - example.Label) * h.LearningRate

		// Accumulate gradients for batch update
		gradientUpdates[example.UserID] = gradientUpdates[example.UserID] + gradient
		gradientUpdates[example.TargetID] = gradientUpdates[example.TargetID] - gradient
	}

	// Apply all gradient updates with lock
	h.mu.Lock()
	defer h.mu.Unlock()
	for nodeID, grad := range gradientUpdates {
		h.updateEmbeddingGradient(nodeID, grad)
	}

	return nil
}

// updateEmbeddingGradient applies a gradient update to a node's embedding
func (h *HybridGNN) updateEmbeddingGradient(nodeID string, gradient float64) {
	node, exists := h.Graph.GetNode(nodeID)
	if !exists || len(node.Embedding) != h.EmbeddingDim {
		return
	}

	// Apply gradient descent to embedding
	for i := range node.Embedding {
		node.Embedding[i] -= gradient * 0.001 // Small step size
	}
}

// GetHybridRecommendations uses the hybrid approach for recommendations
func (h *HybridGNN) GetHybridRecommendations(userID string, targetType NodeType, topK int) ([]*Node, error) {
	userEmbedding, err := h.HybridMessagePassing(userID, 2)
	if err != nil {
		return nil, err
	}

	candidates := h.Graph.GetNodesByType(targetType)
	scores := make([]struct {
		node  *Node
		score float64
	}, 0, len(candidates))

	for _, candidate := range candidates {
		if candidate.ID == userID {
			continue // Skip self
		}

		candidateEmbedding, err := h.HybridMessagePassing(candidate.ID, 1)
		if err != nil {
			continue
		}

		// Hybrid scoring: 60% learned similarity + 40% heuristic features
		learnedSimilarity := h.cosineSimilarity(userEmbedding, candidateEmbedding)

		// Extract direct heuristic score
		heuristicFeatures := h.extractHeuristicFeatures(candidate.ID)
		heuristicScore := 0.0
		for _, feature := range heuristicFeatures {
			heuristicScore += feature
		}
		heuristicScore /= float64(len(heuristicFeatures)) // Average

		// Combine scores
		finalScore := (learnedSimilarity * 0.6) + (heuristicScore * 0.4)

		scores = append(scores, struct {
			node  *Node
			score float64
		}{candidate, finalScore})
	}

	// Sort by score (descending)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return top K
	result := make([]*Node, 0, topK)
	for i := 0; i < len(scores) && i < topK; i++ {
		result = append(result, scores[i].node)
	}

	return result, nil
}

// cosineSimilarity helper for the hybrid GNN
func (h *HybridGNN) cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (normA * normB)
}

// aggregateEmbeddings combines neighbor embeddings for the hybrid GNN
func (h *HybridGNN) aggregateEmbeddings(embeddings [][]float64, weights []float64) []float64 {
	if len(embeddings) == 0 {
		return make([]float64, h.EmbeddingDim)
	}

	result := make([]float64, h.EmbeddingDim)

	// Use mean aggregation for simplicity (can extend to other types)
	totalWeight := 0.0
	for i, embedding := range embeddings {
		weight := weights[i]
		totalWeight += weight
		for j, val := range embedding {
			if j < h.EmbeddingDim {
				result[j] += val * weight
			}
		}
	}

	if totalWeight > 0 {
		for j := range result {
			result[j] /= totalWeight
		}
	}

	return result
}

// GetRecommendations returns ranked recommendations for a user
func (gnn *GNN) GetRecommendations(userID string, targetType NodeType, topK int) ([]*Node, error) {
	userEmbedding, err := gnn.MessagePassing(userID, 2)
	if err != nil {
		return nil, err
	}

	candidates := gnn.Graph.GetNodesByType(targetType)
	scores := make([]struct {
		node  *Node
		score float64
	}, 0, len(candidates))

	for _, candidate := range candidates {
		if candidate.ID == userID {
			continue // Skip self
		}

		candidateEmbedding, err := gnn.MessagePassing(candidate.ID, 1)
		if err != nil {
			continue
		}

		// Combine embedding similarity with engagement and persuasiveness
		embeddingSimilarity := gnn.CosineSimilarity(userEmbedding, candidateEmbedding)

		// Get engagement boost (0.0 to 1.0 normalized)
		engagementBoost := 0.0
		if candidate.Engagement != nil && candidate.Engagement.Score > 0 {
			// Normalize engagement score to 0-1 range using sigmoid
			engagementBoost = 1.0 / (1.0 + math.Exp(-candidate.Engagement.Score/10.0))
		}

		// Get persuasiveness boost (heaviest factor for users)
		persuasivenessBoost := 0.0
		if candidate.Type == UserNode && candidate.Engagement != nil && candidate.Engagement.Persuasiveness > 0 {
			// Normalize persuasiveness to 0-1 range using sigmoid
			persuasivenessBoost = 1.0 / (1.0 + math.Exp(-candidate.Engagement.Persuasiveness/5.0))
		}

		// Get followback health boost for users
		followbackHealthBoost := 1.0
		if candidate.Type == UserNode && candidate.Engagement != nil && candidate.Engagement.FollowbackMetrics != nil {
			followbackHealthBoost = candidate.Engagement.FollowbackMetrics.HealthScore
		}

		// Combine scores with persuasiveness as heaviest factor for users
		var finalScore float64
		if candidate.Type == UserNode {
			// User ranking: 50% persuasiveness + 30% embedding similarity + 20% engagement
			baseUserScore := (persuasivenessBoost * 0.5) + (embeddingSimilarity * 0.3) + (engagementBoost * 0.2)
			// Apply followback health boost to user scores
			finalScore = baseUserScore * followbackHealthBoost
		} else {
			// Non-user ranking: 70% embedding similarity + 30% engagement (original formula)
			finalScore = (embeddingSimilarity * 0.7) + (engagementBoost * 0.3)
		}

		// Apply human vs agent content weighting for non-user content (posts, comments, etc.)
		if candidate.Type != UserNode {
			finalScore = gnn.applyContentAuthorWeighting(candidate, finalScore)
		}

		scores = append(scores, struct {
			node  *Node
			score float64
		}{candidate, finalScore})
	}

	// Sort by score (descending)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return top K
	result := make([]*Node, 0, topK)
	for i := 0; i < len(scores) && i < topK; i++ {
		result = append(result, scores[i].node)
	}

	return result, nil
}

// CosineSimilarity calculates cosine similarity between two embeddings
func (gnn *GNN) CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (normA * normB)
}

// applyContentAuthorWeighting applies human vs agent weighting to content in GNN recommendations
func (gnn *GNN) applyContentAuthorWeighting(contentNode *Node, baseScore float64) float64 {
	// Find the author of this content by looking for 'author' in metadata
	authorID, exists := contentNode.Metadata["author"]
	if !exists {
		return baseScore
	}

	authorIDStr, ok := authorID.(string)
	if !ok {
		return baseScore
	}

	return gnn.Graph.applyContentAuthorWeighting(authorIDStr, baseScore)
}
