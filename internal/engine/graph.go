package engine

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// NodeType represents different types of nodes in the social graph
type NodeType int

const (
	UserNode NodeType = iota
	PostNode
	BusinessNode
	EventNode
	CommentNode
)

// EmotionalReactions tracks different types of emotional responses
type EmotionalReactions struct {
	Like     int `json:"like"`     // x1 - Basic positive signal
	Sympathy int `json:"sympathy"` // x2 - Support, community care
	Humor    int `json:"humor"`    // x3 - Entertaining, engaging
	Anger    int `json:"anger"`    // x4 - High engagement, controversial
	Insight  int `json:"insight"`  // x5 - Knowledge sharing, most valuable
}

// UserSpamMetrics tracks user-level spam indicators
type UserSpamMetrics struct {
	PostCount           int     `json:"post_count"`            // Total posts created by user
	AvgPostEngagement   float64 `json:"avg_post_engagement"`   // Average engagement per post
	FriendToPostRatio   float64 `json:"friend_to_post_ratio"`  // Friends / Posts ratio
	FollowbackRate      float64 `json:"followback_rate"`       // Percentage of mutual follows (0.0-1.0)
	FollowerCount       int     `json:"follower_count"`        // Total followers
	FollowHarvesting    float64 `json:"follow_harvesting"`     // Follow harvesting penalty (0.0-1.0)
	SpamProbability     float64 `json:"spam_probability"`      // 0.0-1.0 spam likelihood
	IsLikelySpammer     bool    `json:"is_likely_spammer"`     // True if spam probability > threshold
}

// FollowbackMetrics tracks followback ratios for user social health
type FollowbackMetrics struct {
	Following      int     `json:"following"`       // Users this user follows
	Followers      int     `json:"followers"`       // Users following this user
	Followbacks    int     `json:"followbacks"`     // Mutual follows (bidirectional)
	FollowbackRate float64 `json:"followback_rate"` // Percentage of mutual follows (0.0 - 1.0)
	HealthScore    float64 `json:"health_score"`    // Social health boost (optimal at 60-70% followback rate)
}

// EngagementMetrics tracks interaction signals for ranking
type EngagementMetrics struct {
	Reactions         EmotionalReactions `json:"reactions"`
	Shares            int                `json:"shares"`
	Bookmarks         int                `json:"bookmarks"`
	Comments          int                `json:"comments"`
	Views             int                `json:"views"`
	Clickthroughs     int                `json:"clickthroughs"`     // User clicks to view full content
	Score             float64            `json:"score"`             // Calculated engagement score
	Persuasiveness    float64            `json:"persuasiveness"`    // User persuasiveness score (for UserNode only)
	SpamScore         float64            `json:"spam_score"`        // Spam/clickbait detection score (higher = more spammy)
	UserSpamFlags     *UserSpamMetrics   `json:"user_spam_flags"`   // User-level spam detection (for UserNode only)
	FollowbackMetrics *FollowbackMetrics `json:"followback_metrics"` // Followback ratio tracking (for UserNode only)
}

// Node represents a node in the social graph
type Node struct {
	ID         string                 `json:"id"`
	Type       NodeType               `json:"type"`
	Embedding  []float64              `json:"embedding"`
	Metadata   map[string]interface{} `json:"metadata"`
	Engagement *EngagementMetrics     `json:"engagement"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

// Edge represents a weighted edge between nodes
type Edge struct {
	From      string    `json:"from"`
	To        string    `json:"to"`
	Weight    float64   `json:"weight"`
	EdgeType  string    `json:"edge_type"` // "friend", "like", "comment", "view", etc.
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Graph represents the social graph with concurrent access support
type Graph struct {
	nodes     map[string]*Node
	edges     map[string]map[string]*Edge // from -> to -> edge
	nodeTypes map[NodeType][]string       // type -> node IDs
	mu        sync.RWMutex
}

// NewGraph creates a new empty graph
func NewGraph() *Graph {
	return &Graph{
		nodes:     make(map[string]*Node),
		edges:     make(map[string]map[string]*Edge),
		nodeTypes: make(map[NodeType][]string),
	}
}

// AddNode adds a node to the graph
func (g *Graph) AddNode(node *Node) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.nodes[node.ID]; exists {
		return fmt.Errorf("node %s already exists", node.ID)
	}

	node.UpdatedAt = time.Now()
	if node.CreatedAt.IsZero() {
		node.CreatedAt = node.UpdatedAt
	}

	// Initialize engagement metrics if not provided
	if node.Engagement == nil {
		node.Engagement = &EngagementMetrics{
			Reactions: EmotionalReactions{
				Like:     0,
				Sympathy: 0,
				Humor:    0,
				Anger:    0,
				Insight:  0,
			},
			Shares:         0,
			Bookmarks:      0,
			Comments:       0,
			Views:          0,
			Clickthroughs:  0,
			Score:          0.0,
			Persuasiveness: 0.0,
			SpamScore:      0.0,
			UserSpamFlags:  nil, // Will be initialized for users only
		}
	}

	g.nodes[node.ID] = node
	g.nodeTypes[node.Type] = append(g.nodeTypes[node.Type], node.ID)

	return nil
}

// GetNode retrieves a node by ID
func (g *Graph) GetNode(id string) (*Node, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	node, exists := g.nodes[id]
	return node, exists
}

// AddEdge adds an edge to the graph
func (g *Graph) AddEdge(edge *Edge) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check if nodes exist
	if _, exists := g.nodes[edge.From]; !exists {
		return fmt.Errorf("source node %s does not exist", edge.From)
	}
	if _, exists := g.nodes[edge.To]; !exists {
		return fmt.Errorf("target node %s does not exist", edge.To)
	}

	// Initialize edge map for source node if needed
	if g.edges[edge.From] == nil {
		g.edges[edge.From] = make(map[string]*Edge)
	}

	edge.UpdatedAt = time.Now()
	if edge.CreatedAt.IsZero() {
		edge.CreatedAt = edge.UpdatedAt
	}

	g.edges[edge.From][edge.To] = edge

	// Update followback metrics if this is a friendship/follow relationship
	if (edge.EdgeType == "friend" || edge.EdgeType == "follow") &&
		g.nodes[edge.From].Type == UserNode && g.nodes[edge.To].Type == UserNode {
		g.updateFollowbackMetrics(edge.From)
		g.updateFollowbackMetrics(edge.To)
	}

	return nil
}

// GetNeighbors returns all neighbors of a node with their edges
func (g *Graph) GetNeighbors(nodeID string) (map[string]*Edge, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	edges, exists := g.edges[nodeID]
	if !exists {
		return make(map[string]*Edge), false
	}

	// Return a copy to avoid concurrent modification
	result := make(map[string]*Edge)
	for k, v := range edges {
		result[k] = v
	}

	return result, true
}

// GetNodesByType returns all nodes of a specific type
func (g *Graph) GetNodesByType(nodeType NodeType) []*Node {
	g.mu.RLock()
	defer g.mu.RUnlock()

	nodeIDs := g.nodeTypes[nodeType]
	nodes := make([]*Node, 0, len(nodeIDs))

	for _, id := range nodeIDs {
		if node, exists := g.nodes[id]; exists {
			nodes = append(nodes, node)
		}
	}

	return nodes
}

// ApplyDecay reduces edge weights over time
func (g *Graph) ApplyDecay(decayRate float64, maxAge time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := time.Now()

	for _, edgeMap := range g.edges {
		for _, edge := range edgeMap {
			age := now.Sub(edge.UpdatedAt)
			if age > maxAge {
				continue
			}

			// Exponential decay: weight * exp(-decay_rate * age_in_hours)
			ageInHours := age.Hours()
			decayFactor := math.Exp(-decayRate * ageInHours)
			edge.Weight *= decayFactor
		}
	}
}

// Stats returns basic graph statistics
func (g *Graph) Stats() map[string]interface{} {
	g.mu.RLock()
	defer g.mu.RUnlock()

	edgeCount := 0
	for _, edgeMap := range g.edges {
		edgeCount += len(edgeMap)
	}

	return map[string]interface{}{
		"nodes": len(g.nodes),
		"edges": edgeCount,
		"node_types": map[string]int{
			"users":      len(g.nodeTypes[UserNode]),
			"posts":      len(g.nodeTypes[PostNode]),
			"businesses": len(g.nodeTypes[BusinessNode]),
			"events":     len(g.nodeTypes[EventNode]),
			"comments":   len(g.nodeTypes[CommentNode]),
		},
	}
}

// UpdateEngagement updates engagement metrics for a node
func (g *Graph) UpdateEngagement(nodeID string, engagementType string, delta int) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	node, exists := g.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node %s not found", nodeID)
	}

	if node.Engagement == nil {
		node.Engagement = &EngagementMetrics{}
	}

	switch engagementType {
	// Emotional reactions
	case "like":
		node.Engagement.Reactions.Like += delta
	case "sympathy":
		node.Engagement.Reactions.Sympathy += delta
	case "humor":
		node.Engagement.Reactions.Humor += delta
	case "anger":
		node.Engagement.Reactions.Anger += delta
	case "insight":
		node.Engagement.Reactions.Insight += delta
	// Other engagement types
	case "share":
		node.Engagement.Shares += delta
	case "bookmark":
		node.Engagement.Bookmarks += delta
	case "comment":
		node.Engagement.Comments += delta
	case "view":
		node.Engagement.Views += delta
	case "clickthrough":
		node.Engagement.Clickthroughs += delta
	default:
		return fmt.Errorf("unknown engagement type: %s", engagementType)
	}

	// Prevent negative values for emotional reactions
	if node.Engagement.Reactions.Like < 0 {
		node.Engagement.Reactions.Like = 0
	}
	if node.Engagement.Reactions.Sympathy < 0 {
		node.Engagement.Reactions.Sympathy = 0
	}
	if node.Engagement.Reactions.Humor < 0 {
		node.Engagement.Reactions.Humor = 0
	}
	if node.Engagement.Reactions.Anger < 0 {
		node.Engagement.Reactions.Anger = 0
	}
	if node.Engagement.Reactions.Insight < 0 {
		node.Engagement.Reactions.Insight = 0
	}
	if node.Engagement.Shares < 0 {
		node.Engagement.Shares = 0
	}
	if node.Engagement.Bookmarks < 0 {
		node.Engagement.Bookmarks = 0
	}
	if node.Engagement.Comments < 0 {
		node.Engagement.Comments = 0
	}
	if node.Engagement.Views < 0 {
		node.Engagement.Views = 0
	}
	if node.Engagement.Clickthroughs < 0 {
		node.Engagement.Clickthroughs = 0
	}

	// Update spam score for posts only
	if node.Type == PostNode {
		node.Engagement.SpamScore = g.calculateSpamScore(node.Engagement)
	}

	// Recalculate engagement score
	node.Engagement.Score = g.calculateEngagementScore(node.Engagement)

	// Update spam flags and persuasiveness score for users only
	if node.Type == UserNode {
		// Initialize user spam tracking if not present
		if node.Engagement.UserSpamFlags == nil {
			node.Engagement.UserSpamFlags = &UserSpamMetrics{}
		}

		// Update user spam detection
		g.updateUserSpamMetrics(node.ID, node.Engagement.UserSpamFlags)

		// Calculate persuasiveness with spam penalty
		node.Engagement.Persuasiveness = g.calculatePersuasivenessScore(node.ID, node.Engagement)
	}

	// For comments, propagate engagement to parent post
	if node.Type == CommentNode {
		if parentID, exists := node.Metadata["parent_id"]; exists {
			if parentIDStr, ok := parentID.(string); ok {
				// Add a portion of comment engagement to parent post
				g.propagateCommentEngagementToPost(parentIDStr, engagementType, delta)
			}
		}
	}

	node.UpdatedAt = time.Now()

	return nil
}

// propagateCommentEngagementToPost adds comment engagement to parent post
func (g *Graph) propagateCommentEngagementToPost(postID string, engagementType string, delta int) {
	postNode, exists := g.nodes[postID]
	if !exists || postNode.Type != PostNode {
		return
	}

	if postNode.Engagement == nil {
		postNode.Engagement = &EngagementMetrics{}
	}

	// Add comment count to post
	if engagementType != "view" && engagementType != "clickthrough" {
		// Comments on comments increase the post's comment count
		postNode.Engagement.Comments += delta
	}

	// Propagate a portion of the comment engagement to the post (30% weight)
	propagatedDelta := int(float64(delta) * 0.3)
	if propagatedDelta > 0 {
		switch engagementType {
		case "like":
			postNode.Engagement.Reactions.Like += propagatedDelta
		case "sympathy":
			postNode.Engagement.Reactions.Sympathy += propagatedDelta
		case "humor":
			postNode.Engagement.Reactions.Humor += propagatedDelta
		case "anger":
			postNode.Engagement.Reactions.Anger += propagatedDelta
		case "insight":
			postNode.Engagement.Reactions.Insight += propagatedDelta
		case "share":
			postNode.Engagement.Shares += propagatedDelta
		case "bookmark":
			postNode.Engagement.Bookmarks += propagatedDelta
		}
	}

	// Recalculate post engagement score
	postNode.Engagement.Score = g.calculateEngagementScore(postNode.Engagement)
	postNode.Engagement.SpamScore = g.calculateSpamScore(postNode.Engagement)
	postNode.UpdatedAt = time.Now()
}

// calculateEngagementScore computes a weighted engagement score with emotional reactions
func (g *Graph) calculateEngagementScore(e *EngagementMetrics) float64 {
	// Emotional reaction scoring with multipliers
	emotionalScore := float64(e.Reactions.Like)*1.0 + // x1 - Basic positive signal
		float64(e.Reactions.Sympathy)*2.0 + // x2 - Support, community care
		float64(e.Reactions.Humor)*3.0 + // x3 - Entertaining, engaging
		float64(e.Reactions.Anger)*4.0 + // x4 - High engagement, controversial
		float64(e.Reactions.Insight)*5.0 // x5 - Knowledge sharing, most valuable

	// Traditional engagement scoring
	traditionalScore := float64(e.Shares)*5.0 + // Shares are viral (equivalent to insight)
		float64(e.Bookmarks)*3.0 + // Bookmarks show strong interest
		float64(e.Comments)*2.0 + // Comments show engagement depth
		float64(e.Clickthroughs)*1.5 + // Clickthroughs show active interest
		float64(e.Views)*0.1 // Views are basic awareness

	// Combine emotional and traditional signals
	totalScore := emotionalScore + traditionalScore

	// Apply spam penalty for posts
	spamPenalty := e.SpamScore * 2.0 // Spam penalty multiplier
	totalScore = math.Max(0.0, totalScore-spamPenalty)

	// Apply logarithmic dampening to prevent runaway scores
	if totalScore > 0 {
		totalScore = math.Log10(totalScore+1) * 10.0
	}

	return totalScore
}

// GetEngagementScore returns the current engagement score for a node
func (g *Graph) GetEngagementScore(nodeID string) (float64, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	node, exists := g.nodes[nodeID]
	if !exists {
		return 0, fmt.Errorf("node %s not found", nodeID)
	}

	if node.Engagement == nil {
		return 0, nil
	}

	return node.Engagement.Score, nil
}

// calculateFollowbackMetrics calculates followback ratios and health scores for a user
func (g *Graph) calculateFollowbackMetrics(userID string) *FollowbackMetrics {
	g.mu.RLock()
	defer g.mu.RUnlock()

	node, exists := g.nodes[userID]
	if !exists || node.Type != UserNode {
		return &FollowbackMetrics{}
	}

	following := 0
	followers := 0
	followbacks := 0

	// Count users this user follows
	if userEdges, exists := g.edges[userID]; exists {
		for _, edge := range userEdges {
			if edge.EdgeType == "friend" || edge.EdgeType == "follow" {
				targetNode, targetExists := g.nodes[edge.To]
				if targetExists && targetNode.Type == UserNode {
					following++

					// Check if target follows back
					if targetEdges, targetFollows := g.edges[edge.To]; targetFollows {
						if backEdge, hasBack := targetEdges[userID]; hasBack {
							if backEdge.EdgeType == "friend" || backEdge.EdgeType == "follow" {
								followbacks++
							}
						}
					}
				}
			}
		}
	}

	// Count users following this user
	for fromUser, userEdges := range g.edges {
		if edge, exists := userEdges[userID]; exists {
			if edge.EdgeType == "friend" || edge.EdgeType == "follow" {
				fromNode, fromExists := g.nodes[fromUser]
				if fromExists && fromNode.Type == UserNode {
					followers++
				}
			}
		}
	}

	// Calculate followback rate (percentage of mutual connections)
	var followbackRate float64
	if following > 0 {
		followbackRate = float64(followbacks) / float64(following)
	}

	// Calculate health score (optimal at 60-70% followback rate)
	healthScore := g.calculateFollowbackHealthScore(followbackRate)

	return &FollowbackMetrics{
		Following:      following,
		Followers:      followers,
		Followbacks:    followbacks,
		FollowbackRate: followbackRate,
		HealthScore:    healthScore,
	}
}

// calculateFollowbackHealthScore calculates social health boost based on followback ratio
func (g *Graph) calculateFollowbackHealthScore(followbackRate float64) float64 {
	if followbackRate < 0.0 {
		return 1.0 // No penalty for edge cases
	}

	// Optimal range: 60-70% followback rate
	if followbackRate >= 0.6 && followbackRate <= 0.7 {
		// Maximum boost in optimal range (1.5x multiplier)
		return 1.5
	}

	// Below 60%: gradual increase from 1.0 to 1.5
	if followbackRate < 0.6 {
		// Linear increase: 1.0 at 0% to 1.5 at 60%
		return 1.0 + (followbackRate / 0.6 * 0.5)
	}

	// Above 70%: maintain 1.5x boost (no penalty)
	return 1.5
}

// calculateFollowHarvestingPenalty detects follow harvesting behavior
func (g *Graph) calculateFollowHarvestingPenalty(followbackRate float64, followerCount int) float64 {
	// Follow harvesting criteria: <20% followback rate AND >100 followers
	if followbackRate < 0.20 && followerCount > 100 {
		// Severe penalty for follow harvesters
		// Scale penalty based on how low the followback rate is
		followHarvestingPenalty := 0.8 // Base 80% penalty

		// Additional penalty for extremely low followback rates
		if followbackRate < 0.10 {
			followHarvestingPenalty = 0.95 // 95% penalty for <10% followback
		} else if followbackRate < 0.15 {
			followHarvestingPenalty = 0.90 // 90% penalty for <15% followback
		}

		// Scale up penalty for massive follower counts (industrial harvesting)
		if followerCount > 500 {
			followHarvestingPenalty = math.Min(1.0, followHarvestingPenalty + 0.1)
		} else if followerCount > 1000 {
			followHarvestingPenalty = 1.0 // Maximum penalty for massive harvesting
		}

		return followHarvestingPenalty
	}

	// No penalty for users with healthy followback rates or reasonable follower counts
	return 0.0
}

// updateFollowbackMetrics updates followback metrics for a user
func (g *Graph) updateFollowbackMetrics(userID string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	node, exists := g.nodes[userID]
	if !exists || node.Type != UserNode {
		return
	}

	if node.Engagement == nil {
		node.Engagement = &EngagementMetrics{}
	}

	// Calculate fresh followback metrics
	metrics := g.calculateFollowbackMetrics(userID)
	node.Engagement.FollowbackMetrics = metrics
	node.UpdatedAt = time.Now()
}

// calculatePersuasivenessScore computes user persuasiveness based on positive contributions and social influence
func (g *Graph) calculatePersuasivenessScore(userID string, e *EngagementMetrics) float64 {
	// Core persuasiveness factors
	positiveContribution := float64(e.Comments) + // Engagement depth
		float64(e.Clickthroughs) + // Active interest shown
		float64(e.Reactions.Insight) // Knowledge sharing

	// Count friends (bidirectional edges of type "friend" or strong connections)
	friends := g.countUserFriends(userID)

	// Social influence multiplier
	socialInfluence := float64(friends)

	// Negative behavior penalty
	toxicityPenalty := float64(e.Reactions.Anger) // Anger reactions reduce persuasiveness

	// Persuasiveness formula: (positive contributions + social influence) - toxicity
	rawScore := positiveContribution + socialInfluence - toxicityPenalty

	// Apply user spam penalty if flagged
	spamPenalty := 0.0
	if e.UserSpamFlags != nil && e.UserSpamFlags.IsLikelySpammer {
		// Heavy penalty for suspected spammers: reduce persuasiveness by spam probability
		spamPenalty = rawScore * e.UserSpamFlags.SpamProbability * 0.8 // Up to 80% reduction
	}

	// Ensure minimum of 0
	persuasivenessScore := math.Max(0.0, rawScore-spamPenalty)

	// Apply square root dampening to prevent runaway scores while maintaining differentiation
	if persuasivenessScore > 0 {
		persuasivenessScore = math.Sqrt(persuasivenessScore) * 3.0 // Scale up for meaningful range
	}

	// Apply followback health boost
	if e.FollowbackMetrics != nil {
		persuasivenessScore *= e.FollowbackMetrics.HealthScore
	}

	// Apply user role multipliers
	persuasivenessScore = g.applyUserRoleMultipliers(userID, persuasivenessScore)

	return persuasivenessScore
}

// applyUserRoleMultipliers applies role-based multipliers to persuasiveness scores
func (g *Graph) applyUserRoleMultipliers(userID string, baseScore float64) float64 {
	node, exists := g.nodes[userID]
	if !exists || node.Type != UserNode {
		return baseScore
	}

	finalScore := baseScore

	// Check for admin role (4x multiplier)
	if role, exists := node.Metadata["role"]; exists && role == "admin" {
		finalScore *= 4.0
	}

	// Check for user type: prioritize humans over agents (0.8x penalty for agents)
	if userType, exists := node.Metadata["user_type"]; exists && userType == "agent" {
		finalScore *= 0.8 // Agents get 20% reduction
	}
	// Humans get no penalty (default 1.0x multiplier)

	return finalScore
}

// applyContentAuthorWeighting applies human vs agent weighting to content (posts/comments)
func (g *Graph) applyContentAuthorWeighting(authorID string, baseScore float64) float64 {
	if authorID == "" {
		return baseScore
	}

	authorNode, exists := g.nodes[authorID]
	if !exists || authorNode.Type != UserNode {
		return baseScore
	}

	// Apply user type weighting: agents get 0.8x weight (20% reduction)
	if userType, exists := authorNode.Metadata["user_type"]; exists && userType == "agent" {
		return baseScore * 0.8 // Prioritize human content over agent content
	}

	// Humans get standard weight (1.0x multiplier)
	return baseScore
}

// countUserFriends counts the number of friendship connections for a user
func (g *Graph) countUserFriends(userID string) int {
	neighbors, exists := g.edges[userID]
	if !exists {
		return 0
	}

	friendCount := 0
	for _, edge := range neighbors {
		// Count friendship edges and strong interest matches
		if edge.EdgeType == "friend" || edge.EdgeType == "strong_interest_match" {
			friendCount++
		}
	}

	return friendCount
}

// GetPersuasivenessScore returns the current persuasiveness score for a user
func (g *Graph) GetPersuasivenessScore(userID string) (float64, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	node, exists := g.nodes[userID]
	if !exists {
		return 0, fmt.Errorf("user %s not found", userID)
	}

	if node.Type != UserNode {
		return 0, fmt.Errorf("persuasiveness score only applies to users")
	}

	if node.Engagement == nil {
		return 0, nil
	}

	return node.Engagement.Persuasiveness, nil
}


// calculateSpamScore detects clickbait/spam content based on engagement patterns
func (g *Graph) calculateSpamScore(e *EngagementMetrics) float64 {
	// Avoid division by zero
	if e.Clickthroughs == 0 {
		return 0.0
	}

	// Calculate meaningful engagement (excludes passive views and clickthroughs)
	meaningfulEngagement := float64(e.Comments + e.Reactions.Like + e.Reactions.Sympathy +
		e.Reactions.Humor + e.Reactions.Anger + e.Reactions.Insight + e.Shares + e.Bookmarks)

	// Calculate clickthrough-to-engagement ratio
	clickthroughRatio := float64(e.Clickthroughs) / math.Max(1.0, meaningfulEngagement)

	// Spam threshold: if clickthroughs are 3x+ higher than meaningful engagement
	spamThreshold := 3.0

	if clickthroughRatio > spamThreshold {
		// Calculate spam score based on how much the ratio exceeds threshold
		excessRatio := clickthroughRatio - spamThreshold
		spamScore := math.Min(10.0, excessRatio) // Cap at 10.0 max spam score

		// Apply minimum clickthrough requirement to avoid false positives
		if e.Clickthroughs >= 5 {
			return spamScore
		}
	}

	return 0.0
}

// GetSpamScore returns the current spam score for a post
func (g *Graph) GetSpamScore(nodeID string) (float64, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	node, exists := g.nodes[nodeID]
	if !exists {
		return 0, fmt.Errorf("node %s not found", nodeID)
	}

	if node.Type != PostNode {
		return 0, fmt.Errorf("spam score only applies to posts")
	}

	if node.Engagement == nil {
		return 0, nil
	}

	return node.Engagement.SpamScore, nil
}

// updateUserSpamMetrics analyzes user posting patterns to detect potential spammers
func (g *Graph) updateUserSpamMetrics(userID string, spamMetrics *UserSpamMetrics) {
	// Count posts created by this user
	postCount := 0
	totalEngagement := 0.0

	for _, node := range g.nodes {
		if node.Type == PostNode && node.Metadata["author"] == userID {
			postCount++
			if node.Engagement != nil {
				totalEngagement += node.Engagement.Score
			}
		}
	}

	// Get friend count
	friendCount := g.countUserFriends(userID)

	// Update metrics
	spamMetrics.PostCount = postCount
	spamMetrics.AvgPostEngagement = 0.0
	if postCount > 0 {
		spamMetrics.AvgPostEngagement = totalEngagement / float64(postCount)
	}

	// Calculate friend-to-post ratio
	spamMetrics.FriendToPostRatio = 0.0
	if postCount > 0 {
		spamMetrics.FriendToPostRatio = float64(friendCount) / float64(postCount)
	}

	// Get followback metrics for follow harvesting detection
	followbackMetrics := g.calculateFollowbackMetrics(userID)
	spamMetrics.FollowbackRate = followbackMetrics.FollowbackRate
	spamMetrics.FollowerCount = followbackMetrics.Followers

	// Calculate follow harvesting penalty
	spamMetrics.FollowHarvesting = g.calculateFollowHarvestingPenalty(followbackMetrics.FollowbackRate, followbackMetrics.Followers)

	// Spam detection algorithm
	spamProbability := 0.0

	// High post volume with few friends = suspicious
	if postCount >= 5 && friendCount < 2 {
		spamProbability += 0.4
	}

	// Very high post volume = more suspicious
	if postCount >= 10 {
		spamProbability += 0.3
	}

	// Low average engagement per post = content not resonating
	if spamMetrics.AvgPostEngagement < 2.0 && postCount >= 3 {
		spamProbability += 0.3
	}

	// Friend-to-post ratio too low = not building community connections
	if spamMetrics.FriendToPostRatio < 0.1 && postCount >= 5 {
		spamProbability += 0.2
	}

	// Follow harvesting penalty - hit them hard!
	// This is the main penalty for users with <20% followback and >100 followers
	spamProbability += spamMetrics.FollowHarvesting

	spamMetrics.SpamProbability = math.Min(1.0, spamProbability)
	spamMetrics.IsLikelySpammer = spamProbability > 0.6 // 60% threshold
}

// GetUserSpamFlags returns spam detection metrics for a user
func (g *Graph) GetUserSpamFlags(userID string) (*UserSpamMetrics, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	node, exists := g.nodes[userID]
	if !exists {
		return nil, fmt.Errorf("user %s not found", userID)
	}

	if node.Type != UserNode {
		return nil, fmt.Errorf("spam flags only apply to users")
	}

	if node.Engagement == nil || node.Engagement.UserSpamFlags == nil {
		return &UserSpamMetrics{}, nil
	}

	return node.Engagement.UserSpamFlags, nil
}

// CommentRankingResult represents a ranked comment with its computed score
type CommentRankingResult struct {
	Comment           *Node   `json:"comment"`
	RankScore         float64 `json:"rank_score"`
	TimeScore         float64 `json:"time_score"`
	EngageScore       float64 `json:"engage_score"`
	AuthorCredibility float64 `json:"author_credibility"` // Author's persuasiveness influence
	IsPromoted        bool    `json:"is_promoted"`        // Auto-upvoted by algorithm
	IsSpamPenalized   bool    `json:"is_spam_penalized"`  // Downvoted due to author spam flags
}

// GetRankedComments returns comments for a post ranked by engagement + time decay
func (g *Graph) GetRankedComments(postID string) ([]*CommentRankingResult, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// Get all comments for this post
	var comments []*Node
	for _, comment := range g.nodeTypes[CommentNode] {
		node := g.nodes[comment]
		if parentID, exists := node.Metadata["parent_id"]; exists && parentID == postID {
			comments = append(comments, node)
		}
	}

	if len(comments) == 0 {
		return []*CommentRankingResult{}, nil
	}

	// Calculate ranking scores for each comment
	results := make([]*CommentRankingResult, len(comments))
	now := time.Now()

	for i, comment := range comments {
		// Time decay score (newer = higher, but decays over time)
		hoursSincePosted := now.Sub(comment.CreatedAt).Hours()
		timeScore := math.Max(0.1, 1.0/(1.0+hoursSincePosted/24.0)) // Decays over days

		// Engagement score (normalized 0-1)
		engageScore := 0.0
		if comment.Engagement != nil && comment.Engagement.Score > 0 {
			// Sigmoid normalization for comments (tighter than posts)
			engageScore = 1.0 / (1.0 + math.Exp(-comment.Engagement.Score/5.0))
		}

		// Author credibility score based on persuasiveness
		authorCredibility := 0.5 // Default neutral credibility
		authorID := ""
		if authorIDVal, exists := comment.Metadata["author"]; exists {
			authorID = authorIDVal.(string)
			if authorNode, authorExists := g.nodes[authorID]; authorExists && authorNode.Type == UserNode {
				if authorNode.Engagement != nil && authorNode.Engagement.Persuasiveness > 0 {
					// Normalize persuasiveness to 0-1 range for credibility
					authorCredibility = math.Min(1.0, authorNode.Engagement.Persuasiveness/20.0)
				}
			}
		}

		// Check for spam penalties
		isSpamPenalized := false
		spamPenalty := 1.0 // Default no penalty
		if authorID != "" {
			if authorNode, exists := g.nodes[authorID]; exists && authorNode.Type == UserNode {
				if authorNode.Engagement != nil && authorNode.Engagement.UserSpamFlags != nil && authorNode.Engagement.UserSpamFlags.IsLikelySpammer {
					spamPenalty = 1.0 - (authorNode.Engagement.UserSpamFlags.SpamProbability * 0.7) // Up to 70% penalty
					isSpamPenalized = true
				}
			}
		}

		// Enhanced ranking: 30% time + 40% engagement + 30% author credibility
		baseScore := (timeScore * 0.3) + (engageScore * 0.4) + (authorCredibility * 0.3)
		rankScore := baseScore * spamPenalty // Apply spam penalty

		// Apply human vs agent content weight adjustment
		rankScore = g.applyContentAuthorWeighting(authorID, rankScore)

		// Auto-upvote logic: high engagement + high credibility = promotion
		isPromoted := false
		if engageScore > 0.6 && authorCredibility > 0.7 && !isSpamPenalized {
			rankScore *= 1.4 // 40% boost for credible high-quality comments
			isPromoted = true
		}

		results[i] = &CommentRankingResult{
			Comment:           comment,
			RankScore:         rankScore,
			TimeScore:         timeScore,
			EngageScore:       engageScore,
			AuthorCredibility: authorCredibility,
			IsPromoted:        isPromoted,
			IsSpamPenalized:   isSpamPenalized,
		}
	}

	// Sort by rank score (descending)
	// This will surface high-engagement comments while maintaining some chronological order
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].RankScore < results[j].RankScore {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	return results, nil
}