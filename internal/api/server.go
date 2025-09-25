package api

import (
	"encoding/json"
	"io"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"

	"github.com/LynnColeArt/socialgnn/internal/engine"
)

// Server handles HTTP API requests
type Server struct {
	engine        *engine.Engine
	interestGraph *engine.InterestGraph
}

// NewServer creates a new API server
func NewServer(e *engine.Engine) *Server {
	return &Server{
		engine:        e,
		interestGraph: engine.NewInterestGraph(e),
	}
}

// Router returns the configured HTTP router
func (s *Server) Router() *mux.Router {
	r := mux.NewRouter()

	// API routes
	api := r.PathPrefix("/api").Subrouter()
	api.HandleFunc("/nodes", s.addNode).Methods("POST")
	api.HandleFunc("/edges", s.addEdge).Methods("POST")
	api.HandleFunc("/recommendations/{userID}", s.getRecommendations).Methods("GET")
	api.HandleFunc("/similarity/{user1}/{user2}", s.getSimilarity).Methods("GET")
	api.HandleFunc("/stats", s.getStats).Methods("GET")

	// Interest graph routes
	api.HandleFunc("/interest-graph/build", s.buildInterestGraph).Methods("POST")
	api.HandleFunc("/interest-graph/connections/{userID}", s.getUserConnections).Methods("GET")
	api.HandleFunc("/interest-graph/load-sample", s.loadSampleInterestData).Methods("POST")

	// Engagement tracking routes
	api.HandleFunc("/engagement/{nodeID}", s.updateEngagement).Methods("POST")
	api.HandleFunc("/engagement/{nodeID}", s.getEngagementScore).Methods("GET")

	// Persuasiveness routes
	api.HandleFunc("/persuasiveness/{userID}", s.getPersuasivenessScore).Methods("GET")

	// Spam detection routes
	api.HandleFunc("/spam/{postID}", s.getSpamScore).Methods("GET")
	api.HandleFunc("/user-spam/{userID}", s.getUserSpamFlags).Methods("GET")

	// User followback metrics routes
	api.HandleFunc("/followback/{userID}", s.getFollowbackMetrics).Methods("GET")

	// Comment ranking routes
	api.HandleFunc("/comments/{postID}", s.getRankedComments).Methods("GET")

	// Training routes
	api.HandleFunc("/train/example", s.addTrainingExample).Methods("POST")
	api.HandleFunc("/train/batch", s.trainBatch).Methods("POST")

	// Health check
	r.HandleFunc("/health", s.health).Methods("GET")

	// CORS middleware
	r.Use(corsMiddleware)

	return r
}

// Request/Response types
type AddNodeRequest struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Metadata map[string]interface{} `json:"metadata"`
}

type AddEdgeRequest struct {
	From     string  `json:"from"`
	To       string  `json:"to"`
	Weight   float64 `json:"weight"`
	EdgeType string  `json:"edge_type"`
}

type RecommendationsResponse struct {
	Recommendations []NodeResponse `json:"recommendations"`
}

type NodeResponse struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Metadata map[string]interface{} `json:"metadata"`
}

type SimilarityResponse struct {
	User1      string  `json:"user1"`
	User2      string  `json:"user2"`
	Similarity float64 `json:"similarity"`
}

type EngagementRequest struct {
	Type  string `json:"type"`  // "like", "share", "bookmark", "comment", "view"
	Delta int    `json:"delta"` // +1 or -1
}

type EngagementResponse struct {
	NodeID string  `json:"node_id"`
	Score  float64 `json:"score"`
	Status string  `json:"status"`
}

type PersuasivenessResponse struct {
	UserID         string  `json:"user_id"`
	Persuasiveness float64 `json:"persuasiveness"`
	Status         string  `json:"status"`
}

type SpamResponse struct {
	PostID    string  `json:"post_id"`
	SpamScore float64 `json:"spam_score"`
	Status    string  `json:"status"`
	IsSpam    bool    `json:"is_spam"` // True if spam score > threshold
}

type UserSpamResponse struct {
	UserID            string  `json:"user_id"`
	PostCount         int     `json:"post_count"`
	AvgPostEngagement float64 `json:"avg_post_engagement"`
	FriendToPostRatio float64 `json:"friend_to_post_ratio"`
	FollowbackRate    float64 `json:"followback_rate"`
	FollowerCount     int     `json:"follower_count"`
	FollowHarvesting  float64 `json:"follow_harvesting"`
	SpamProbability   float64 `json:"spam_probability"`
	IsLikelySpammer   bool    `json:"is_likely_spammer"`
	Status            string  `json:"status"`
}

// Handlers
func (s *Server) addNode(w http.ResponseWriter, r *http.Request) {
	var req AddNodeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	nodeType := parseNodeType(req.Type)
	if nodeType == -1 {
		http.Error(w, "Invalid node type", http.StatusBadRequest)
		return
	}

	if err := s.engine.AddNode(req.ID, nodeType, req.Metadata); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"status": "created"})
}

func (s *Server) addEdge(w http.ResponseWriter, r *http.Request) {
	var req AddEdgeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if err := s.engine.AddEdge(req.From, req.To, req.Weight, req.EdgeType); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"status": "created"})
}

func (s *Server) getRecommendations(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userID"]

	// Parse query parameters
	nodeType := engine.PostNode // default
	if t := r.URL.Query().Get("type"); t != "" {
		nodeType = parseNodeType(t)
		if nodeType == -1 {
			http.Error(w, "Invalid node type", http.StatusBadRequest)
			return
		}
	}

	limit := 10 // default
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil {
			limit = parsed
		}
	}

	recommendations, err := s.engine.GetRecommendations(userID, nodeType, limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := RecommendationsResponse{
		Recommendations: make([]NodeResponse, len(recommendations)),
	}

	for i, rec := range recommendations {
		response.Recommendations[i] = NodeResponse{
			ID:       rec.ID,
			Type:     nodeTypeToString(rec.Type),
			Metadata: rec.Metadata,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) getSimilarity(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	user1 := vars["user1"]
	user2 := vars["user2"]

	similarity, err := s.engine.GetSimilarity(user1, user2)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := SimilarityResponse{
		User1:      user1,
		User2:      user2,
		Similarity: similarity,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) getStats(w http.ResponseWriter, r *http.Request) {
	stats := s.engine.GetStats()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "healthy",
		"service": "socialgnn-api",
	})
}

// Helper functions
func parseNodeType(t string) engine.NodeType {
	switch t {
	case "user":
		return engine.UserNode
	case "post":
		return engine.PostNode
	case "business":
		return engine.BusinessNode
	case "event":
		return engine.EventNode
	case "comment":
		return engine.CommentNode
	default:
		return -1
	}
}

func nodeTypeToString(t engine.NodeType) string {
	switch t {
	case engine.UserNode:
		return "user"
	case engine.PostNode:
		return "post"
	case engine.BusinessNode:
		return "business"
	case engine.EventNode:
		return "event"
	case engine.CommentNode:
		return "comment"
	default:
		return "unknown"
	}
}

// Interest Graph API handlers

func (s *Server) buildInterestGraph(w http.ResponseWriter, r *http.Request) {
	var profiles []engine.UserProfile
	if err := json.NewDecoder(r.Body).Decode(&profiles); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if err := s.interestGraph.BuildInterestConnections(profiles); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"status":   "success",
		"message":  "Interest graph built successfully",
		"profiles": len(profiles),
		"stats":    s.engine.GetStats(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) getUserConnections(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userID"]

	// Get all connections for this user from the graph
	neighbors, exists := s.engine.GetNeighbors(userID)
	if !exists {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	type ConnectionResponse struct {
		UserID   string  `json:"user_id"`
		Weight   float64 `json:"weight"`
		EdgeType string  `json:"edge_type"`
		Created  string  `json:"created"`
	}

	connections := make([]ConnectionResponse, 0, len(neighbors))
	for neighborID, edge := range neighbors {
		connections = append(connections, ConnectionResponse{
			UserID:   neighborID,
			Weight:   edge.Weight,
			EdgeType: edge.EdgeType,
			Created:  edge.CreatedAt.Format("2006-01-02T15:04:05Z"),
		})
	}

	response := map[string]interface{}{
		"user_id":     userID,
		"connections": connections,
		"total":       len(connections),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) loadSampleInterestData(w http.ResponseWriter, r *http.Request) {
	// Load sample Riverside profiles
	profiles := s.interestGraph.LoadRiversideInterestProfiles()

	// Add users to graph if they don't exist
	for _, profile := range profiles {
		userMetadata := map[string]interface{}{
			"name":          profile.UserID,
			"age":           profile.Age,
			"occupation":    profile.Occupation,
			"joined_at":     profile.JoinedAt,
			"interests":     profile.Interests,
			"location":      profile.Location,
			"activity_tags": profile.ActivityTags,
		}

		// Try to add user (will skip if already exists)
		s.engine.AddNode(profile.UserID, engine.UserNode, userMetadata)
	}

	// Build interest connections
	if err := s.interestGraph.BuildInterestConnections(profiles); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"status":   "success",
		"message":  "Sample interest data loaded successfully",
		"profiles": len(profiles),
		"stats":    s.engine.GetStats(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Engagement API handlers

func (s *Server) updateEngagement(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	nodeID := vars["nodeID"]

	var req EngagementRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Validate engagement type (including emotional reactions)
	validTypes := map[string]bool{
		// Emotional reactions
		"like": true, "sympathy": true, "humor": true, "anger": true, "insight": true,
		// Traditional engagement
		"share": true, "bookmark": true, "comment": true, "view": true, "clickthrough": true,
	}
	if !validTypes[req.Type] {
		http.Error(w, "Invalid engagement type", http.StatusBadRequest)
		return
	}

	// Update engagement
	if err := s.engine.UpdateEngagement(nodeID, req.Type, req.Delta); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Get updated score
	score, err := s.engine.GetEngagementScore(nodeID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := EngagementResponse{
		NodeID: nodeID,
		Score:  score,
		Status: "updated",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) getEngagementScore(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	nodeID := vars["nodeID"]

	score, err := s.engine.GetEngagementScore(nodeID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := EngagementResponse{
		NodeID: nodeID,
		Score:  score,
		Status: "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Persuasiveness API handlers

func (s *Server) getPersuasivenessScore(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userID"]

	score, err := s.engine.GetPersuasivenessScore(userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := PersuasivenessResponse{
		UserID:         userID,
		Persuasiveness: score,
		Status:         "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Spam detection API handlers

func (s *Server) getSpamScore(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	postID := vars["postID"]

	score, err := s.engine.GetSpamScore(postID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Spam threshold: score > 1.0 is considered spam
	isSpam := score > 1.0

	response := SpamResponse{
		PostID:    postID,
		SpamScore: score,
		Status:    "success",
		IsSpam:    isSpam,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) getUserSpamFlags(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userID"]

	spamFlags, err := s.engine.GetUserSpamFlags(userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := UserSpamResponse{
		UserID:            userID,
		PostCount:         spamFlags.PostCount,
		AvgPostEngagement: spamFlags.AvgPostEngagement,
		FriendToPostRatio: spamFlags.FriendToPostRatio,
		FollowbackRate:    spamFlags.FollowbackRate,
		FollowerCount:     spamFlags.FollowerCount,
		FollowHarvesting:  spamFlags.FollowHarvesting,
		SpamProbability:   spamFlags.SpamProbability,
		IsLikelySpammer:   spamFlags.IsLikelySpammer,
		Status:            "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) getFollowbackMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userID"]

	metrics, err := s.engine.GetFollowbackMetrics(userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// Comment ranking API handlers

func (s *Server) getRankedComments(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	postID := vars["postID"]

	rankedComments, err := s.engine.GetRankedComments(postID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"post_id":         postID,
		"ranked_comments": rankedComments,
		"total":           len(rankedComments),
		"algorithm":       "engagement_boost + time_decay",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Training handlers
type TrainingExampleRequest struct {
	UserID string  `json:"user_id"`
	ItemID string  `json:"item_id"`
	Rating float64 `json:"rating"`
}

type TrainBatchRequest struct {
	Epochs    int `json:"epochs"`
	BatchSize int `json:"batch_size"`
}

type TrainBatchResponse struct {
	Loss     float64 `json:"loss"`
	Epochs   int     `json:"epochs_completed"`
	Examples int     `json:"training_examples"`
}

func (s *Server) addTrainingExample(w http.ResponseWriter, r *http.Request) {
	var req TrainingExampleRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if err := s.engine.AddTrainingExample(req.UserID, req.ItemID, req.Rating); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "example_added"})
}

func (s *Server) trainBatch(w http.ResponseWriter, r *http.Request) {
	req := TrainBatchRequest{Epochs: 1, BatchSize: 32}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if err != io.EOF {
			// Fallback to defaults for backward compatibility when body is empty or invalid
			req.Epochs = 1
			req.BatchSize = 32
		}
	}
	if req.Epochs <= 0 {
		req.Epochs = 1
	}
	if req.BatchSize <= 0 {
		req.BatchSize = 32
	}

	loss, err := s.engine.TrainBatch(req.Epochs, req.BatchSize)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := TrainBatchResponse{
		Loss:     loss,
		Epochs:   req.Epochs,
		Examples: s.engine.GetTrainingExampleCount(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CORS middleware
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}
