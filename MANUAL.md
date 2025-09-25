# SocialGNN User Manual

Complete guide for using the SocialGNN system for social media ranking and recommendations.

## Table of Contents

- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Data Models](#data-models)
- [Integration Examples](#integration-examples)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Server Startup

```bash
# Start with sample data (recommended for testing)
LOAD_SAMPLE_DATA=true go run cmd/server/main.go

# Start production server
PORT=8080 go run cmd/server/main.go

# Docker deployment
docker-compose up -d
```

### Verify Installation

```bash
# Check server health
curl http://localhost:8080/health

# View graph statistics
curl http://localhost:8080/api/stats
```

Expected response:
```json
{
  "total_nodes": 4,
  "total_edges": 6,
  "node_types": {
    "user": 4,
    "post": 3,
    "comment": 0
  },
  "status": "healthy"
}
```

## API Reference

### Authentication

Currently, the API doesn't require authentication. In production, add JWT or API key authentication before the handlers.

### Core Node Management

#### Add a User Node

```bash
POST /api/nodes
Content-Type: application/json

{
  "id": "user_sarah",
  "type": "user",
  "metadata": {
    "name": "Sarah Mitchell",
    "role": "admin",        // "admin" | "user"
    "user_type": "human",   // "human" | "agent"
    "email": "sarah@example.com",
    "interests": ["photography", "hiking", "nature"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "node_id": "user_sarah"
}
```

#### Add a Post Node

```bash
POST /api/nodes
Content-Type: application/json

{
  "id": "post_123",
  "type": "post",
  "metadata": {
    "content": "Beautiful sunset over the mountains! 🌄",
    "author": "user_sarah",
    "category": "nature",
    "hashtags": ["sunset", "mountains", "photography"],
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

#### Add a Comment Node

```bash
POST /api/nodes
Content-Type: application/json

{
  "id": "comment_456",
  "type": "comment",
  "metadata": {
    "content": "Amazing photo! What camera did you use?",
    "author": "user_jake",
    "parent_id": "post_123",
    "created_at": "2024-01-15T11:15:00Z"
  }
}
```

### Relationship Management

#### Create User-to-User Relationships

```bash
POST /api/edges
Content-Type: application/json

{
  "from": "user_sarah",
  "to": "user_jake",
  "weight": 0.8,
  "edge_type": "friend"    // "friend" | "follow"
}
```

#### Create Engagement Relationships

```bash
# User likes post
POST /api/edges
{
  "from": "user_jake",
  "to": "post_123",
  "weight": 0.7,
  "edge_type": "like"
}

# User shares post
POST /api/edges
{
  "from": "user_emma",
  "to": "post_123",
  "weight": 0.9,
  "edge_type": "share"
}

# User bookmarks post
POST /api/edges
{
  "from": "user_tom",
  "to": "post_123",
  "weight": 0.6,
  "edge_type": "bookmark"
}
```

#### Create Author Relationships

```bash
# Connect author to their content
POST /api/edges
{
  "from": "user_sarah",
  "to": "post_123",
  "weight": 1.0,
  "edge_type": "author"
}
```

### Getting Recommendations

#### Get Post Recommendations for User

```bash
GET /api/recommendations/user_jake/post?limit=10
```

**Response:**
```json
{
  "user_id": "user_jake",
  "recommendations": [
    {
      "id": "post_123",
      "type": "post",
      "metadata": {
        "content": "Beautiful sunset...",
        "author": "user_sarah",
        "category": "nature"
      },
      "engagement": {
        "score": 15.2,
        "reactions": {"like": 5, "love": 2, "insight": 1},
        "comments": 3,
        "shares": 1
      },
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T12:00:00Z"
    }
  ],
  "total_count": 1,
  "status": "success"
}
```

#### Get User Recommendations (People to Follow)

```bash
GET /api/recommendations/user_jake/user?limit=5
```

**Response:**
```json
{
  "user_id": "user_jake",
  "recommendations": [
    {
      "id": "user_emma",
      "type": "user",
      "metadata": {
        "name": "Emma Chen",
        "role": "user",
        "interests": ["technology", "cooking", "travel"]
      },
      "engagement": {
        "persuasiveness": 8.4,
        "followback_metrics": {
          "following": 120,
          "followers": 85,
          "followback_rate": 0.68,
          "health_score": 1.45
        }
      }
    }
  ]
}
```

### Engagement Management

#### Update Post Engagement

```bash
PUT /api/engagement/post_123
Content-Type: application/json

{
  "engagement_type": "like",
  "delta": 1                // +1 like
}

# Other engagement types:
# "like", "love", "laugh", "angry", "sad", "insight"
# "share", "bookmark", "comment", "view", "clickthrough"
```

#### Batch Update Engagement

```bash
PUT /api/engagement/post_123
{
  "updates": [
    {"engagement_type": "like", "delta": 3},
    {"engagement_type": "share", "delta": 1},
    {"engagement_type": "comment", "delta": 2}
  ]
}
```

### Analytics and Monitoring

#### Get Spam Score for Content

```bash
GET /api/spam/post_123
```

**Response:**
```json
{
  "post_id": "post_123",
  "spam_score": 0.15,
  "is_spam": false,
  "factors": {
    "engagement_ratio": 0.85,
    "author_credibility": 0.9,
    "content_quality": 0.8
  },
  "status": "success"
}
```

#### Get User Spam Flags

```bash
GET /api/user-spam/user_suspicious
```

**Response:**
```json
{
  "user_id": "user_suspicious",
  "post_count": 25,
  "avg_post_engagement": 1.2,
  "friend_to_post_ratio": 0.08,
  "followback_rate": 0.15,
  "follower_count": 1250,
  "follow_harvesting": 0.85,
  "spam_probability": 0.92,
  "is_likely_spammer": true,
  "status": "success"
}
```

#### Get Followback Metrics

```bash
GET /api/followback/user_sarah
```

**Response:**
```json
{
  "following": 45,
  "followers": 32,
  "followbacks": 28,
  "followback_rate": 0.62,
  "health_score": 1.48
}
```

#### Get Ranked Comments

```bash
GET /api/comments/post_123
```

**Response:**
```json
{
  "post_id": "post_123",
  "comments": [
    {
      "comment": {
        "id": "comment_789",
        "metadata": {
          "content": "This is incredibly insightful!",
          "author": "user_expert"
        }
      },
      "rank_score": 0.95,
      "time_score": 0.8,
      "engage_score": 0.9,
      "author_credibility": 1.2,
      "is_promoted": true,
      "is_spam_penalized": false
    }
  ]
}
```

### Training Endpoints

#### Add Training Example

```bash
POST /api/train/example
Content-Type: application/json

{
  "user_id": "user_sarah",
  "item_id": "post_123",
  "rating": 1
}
```

**Response:**
```json
{
  "status": "example_added"
}
```

#### Run Mini-Batch Training

```bash
POST /api/train/batch
Content-Type: application/json

{
  "epochs": 3,
  "batch_size": 64
}
```

- Both `epochs` and `batch_size` are optional and default to `1` and `32` respectively when omitted or invalid.
- The endpoint returns HTTP `200` with the number of examples processed once at least one training example exists.
- If no training data has been submitted yet, the endpoint returns HTTP `400` with an error payload instead of a server error.

**Response:**
```json
{
  "loss": 0.1,
  "epochs_completed": 3,
  "training_examples": 42
}
```

**Error (no training examples yet):**
```json
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": "no training data available"
}
```

### Admin and Statistics

#### Get Graph Statistics

```bash
GET /api/stats
```

**Response:**
```json
{
  "total_nodes": 1250,
  "total_edges": 8450,
  "node_types": {
    "user": 500,
    "post": 650,
    "comment": 100
  },
  "edge_types": {
    "friend": 1200,
    "like": 4500,
    "share": 800,
    "comment": 1000,
    "author": 750
  },
  "avg_degree": 6.76,
  "memory_usage_mb": 145,
  "status": "healthy"
}
```

#### Get User Similarity

```bash
GET /api/similarity/user_sarah/user_emma
```

**Response:**
```json
{
  "user1_id": "user_sarah",
  "user2_id": "user_emma",
  "similarity_score": 0.73,
  "common_interests": ["photography", "travel"],
  "status": "success"
}
```

#### Get User Persuasiveness Score

```bash
GET /api/persuasiveness/user_sarah
```

**Response:**
```json
{
  "user_id": "user_sarah",
  "persuasiveness_score": 9.2,
  "factors": {
    "positive_contributions": 45,
    "social_influence": 32,
    "role_multiplier": 4.0,
    "followback_health": 1.48
  },
  "status": "success"
}
```

## Data Models

### Node Types

#### UserNode
```go
type User struct {
    ID       string                 `json:"id"`
    Type     NodeType              `json:"type"` // "user"
    Metadata map[string]interface{} `json:"metadata"`
    // Metadata fields:
    // - name: string
    // - role: "admin" | "user"
    // - user_type: "human" | "agent"
    // - email: string
    // - interests: []string
}
```

#### PostNode
```go
type Post struct {
    ID       string                 `json:"id"`
    Type     NodeType              `json:"type"` // "post"
    Metadata map[string]interface{} `json:"metadata"`
    // Metadata fields:
    // - content: string
    // - author: string (user ID)
    // - category: string
    // - hashtags: []string
    // - created_at: time
}
```

#### CommentNode
```go
type Comment struct {
    ID       string                 `json:"id"`
    Type     NodeType              `json:"type"` // "comment"
    Metadata map[string]interface{} `json:"metadata"`
    // Metadata fields:
    // - content: string
    // - author: string (user ID)
    // - parent_id: string (post ID)
    // - created_at: time
}
```

### Edge Types

- **friend**: Bidirectional friendship (weight: 0.8-1.0)
- **follow**: Unidirectional following (weight: 0.5-0.8)
- **like**: User likes content (weight: 0.7)
- **love**: Strong positive reaction (weight: 0.9)
- **share**: User shares content (weight: 0.9)
- **bookmark**: User bookmarks content (weight: 0.6)
- **comment**: User comments on content (weight: 0.8)
- **author**: User authors content (weight: 1.0)

### Engagement Metrics

```go
type EngagementMetrics struct {
    Reactions      EmotionalReactions `json:"reactions"`
    Shares         int                `json:"shares"`
    Bookmarks      int                `json:"bookmarks"`
    Comments       int                `json:"comments"`
    Views          int                `json:"views"`
    Clickthroughs  int                `json:"clickthroughs"`
    Score          float64            `json:"score"`
    Persuasiveness float64            `json:"persuasiveness"`
    SpamScore      float64            `json:"spam_score"`
}

type EmotionalReactions struct {
    Like    int `json:"like"`    // x1 multiplier
    Love    int `json:"love"`    // x2 multiplier
    Laugh   int `json:"laugh"`   // x1 multiplier
    Angry   int `json:"angry"`   // -1 multiplier
    Sad     int `json:"sad"`     // -0.5 multiplier
    Insight int `json:"insight"` // x5 multiplier (most valuable)
}
```

## Integration Examples

### MongoDB Integration Service

```go
package main

import (
    "context"
    "log"

    "go.mongodb.org/mongo-driver/mongo"
    "go.mongodb.org/mongo-driver/mongo/options"
    "github.com/LynnColeArt/socialgnn/pkg/engine"
)

type SocialService struct {
    db        *mongo.Database
    socialGNN *engine.Engine
}

func NewSocialService(mongoURI, dbName string) (*SocialService, error) {
    // Connect to MongoDB
    client, err := mongo.Connect(context.TODO(), options.Client().ApplyURI(mongoURI))
    if err != nil {
        return nil, err
    }

    // Initialize SocialGNN
    socialGNN := engine.NewEngine()

    return &SocialService{
        db:        client.Database(dbName),
        socialGNN: socialGNN,
    }, nil
}

func (s *SocialService) SyncUserFromMongo(userID string) error {
    var user User
    err := s.db.Collection("users").FindOne(context.TODO(), bson.M{"_id": userID}).Decode(&user)
    if err != nil {
        return err
    }

    metadata := map[string]interface{}{
        "name":      user.Name,
        "role":      user.Role,
        "user_type": user.UserType,
        "interests": user.Interests,
    }

    return s.socialGNN.AddNode(user.ID, engine.UserNode, metadata)
}

func (s *SocialService) GetPersonalizedFeed(userID string, limit int) ([]*Post, error) {
    // Get recommendations from GNN
    recommendations, err := s.socialGNN.GetRecommendations(userID, engine.PostNode, limit*2)
    if err != nil {
        return s.getFallbackFeed(userID, limit)
    }

    // Fetch full post data from MongoDB
    var posts []*Post
    for _, node := range recommendations {
        var post Post
        err := s.db.Collection("posts").FindOne(context.TODO(), bson.M{"_id": node.ID}).Decode(&post)
        if err == nil {
            posts = append(posts, &post)
        }

        if len(posts) >= limit {
            break
        }
    }

    return posts, nil
}

func (s *SocialService) HandleUserLike(userID, postID string) error {
    // Update MongoDB
    _, err := s.db.Collection("likes").InsertOne(context.TODO(), Like{
        UserID:    userID,
        PostID:    postID,
        CreatedAt: time.Now(),
    })
    if err != nil {
        return err
    }

    // Update GNN graph
    s.socialGNN.AddEdge(userID, postID, 0.7, "like")
    s.socialGNN.UpdateEngagement(postID, "like", 1)

    return nil
}

func (s *SocialService) getFallbackFeed(userID string, limit int) ([]*Post, error) {
    // Fallback to chronological or simple ranking
    cursor, err := s.db.Collection("posts").Find(
        context.TODO(),
        bson.M{},
        options.Find().SetSort(bson.M{"created_at": -1}).SetLimit(int64(limit)),
    )
    if err != nil {
        return nil, err
    }

    var posts []*Post
    if err = cursor.All(context.TODO(), &posts); err != nil {
        return nil, err
    }

    return posts, nil
}
```

### Express.js API Integration

```javascript
const express = require('express');
const axios = require('axios');

const app = express();
const SOCIALGNN_URL = 'http://localhost:8080';

// Get personalized feed
app.get('/api/feed/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const limit = req.query.limit || 20;

    // Get recommendations from SocialGNN
    const response = await axios.get(`${SOCIALGNN_URL}/api/recommendations/${userId}/post?limit=${limit}`);
    const recommendations = response.data.recommendations;

    // Fetch full post data from your database
    const posts = await Promise.all(
      recommendations.map(async (rec) => {
        const post = await Post.findById(rec.id);
        return {
          ...post.toObject(),
          gnn_score: rec.engagement?.score || 0
        };
      })
    );

    res.json({ posts });
  } catch (error) {
    // Fallback to chronological feed
    const posts = await Post.find()
      .sort({ createdAt: -1 })
      .limit(parseInt(req.query.limit) || 20);
    res.json({ posts });
  }
});

// Handle user interactions
app.post('/api/posts/:postId/like', async (req, res) => {
  const { postId } = req.params;
  const { userId } = req.body;

  try {
    // Update your database
    await Like.create({ userId, postId });

    // Update SocialGNN
    await axios.post(`${SOCIALGNN_URL}/api/edges`, {
      from: userId,
      to: postId,
      weight: 0.7,
      edge_type: 'like'
    });

    await axios.put(`${SOCIALGNN_URL}/api/engagement/${postId}`, {
      engagement_type: 'like',
      delta: 1
    });

    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

## Advanced Usage

### Custom Training Pipeline

```go
func (s *SocialService) TrainOnUserInteractions() error {
    // Collect recent interactions from MongoDB
    cursor, err := s.db.Collection("interactions").Find(
        context.TODO(),
        bson.M{"created_at": bson.M{"$gte": time.Now().Add(-24 * time.Hour)}},
    )
    if err != nil {
        return err
    }

    var interactions []Interaction
    if err = cursor.All(context.TODO(), &interactions); err != nil {
        return err
    }

    // Add training examples to GNN
    gnn := s.socialGNN.(*engine.GNN)
    if gnn.HybridGNN != nil {
        for _, interaction := range interactions {
            label := 1.0 // Positive interaction
            if interaction.Type == "unlike" || interaction.Type == "report" {
                label = 0.0 // Negative interaction
            }

            gnn.HybridGNN.AddTrainingExample(
                interaction.UserID,
                interaction.TargetID,
                interaction.Type,
                label,
            )
        }

        // Train on batch
        return gnn.HybridGNN.TrainOnBatch(100)
    }

    return nil
}
```

### Background Sync Service

```go
func (s *SocialService) StartBackgroundSync() {
    ticker := time.NewTicker(5 * time.Minute)
    go func() {
        for range ticker.C {
            s.syncRecentChanges()
            s.TrainOnUserInteractions()
        }
    }()
}

func (s *SocialService) syncRecentChanges() {
    // Sync users modified in last 5 minutes
    cursor, _ := s.db.Collection("users").Find(
        context.TODO(),
        bson.M{"updated_at": bson.M{"$gte": time.Now().Add(-5 * time.Minute)}},
    )

    var users []User
    cursor.All(context.TODO(), &users)

    for _, user := range users {
        s.SyncUserFromMongo(user.ID)
    }
}
```

### Caching Layer

```go
import "github.com/go-redis/redis/v8"

type CachedSocialService struct {
    *SocialService
    redis *redis.Client
}

func (c *CachedSocialService) GetPersonalizedFeed(userID string, limit int) ([]*Post, error) {
    // Check cache first
    cacheKey := fmt.Sprintf("feed:%s:%d", userID, limit)
    cached, err := c.redis.Get(context.TODO(), cacheKey).Result()
    if err == nil {
        var posts []*Post
        json.Unmarshal([]byte(cached), &posts)
        return posts, nil
    }

    // Get from GNN
    posts, err := c.SocialService.GetPersonalizedFeed(userID, limit)
    if err != nil {
        return nil, err
    }

    // Cache for 5 minutes
    data, _ := json.Marshal(posts)
    c.redis.Set(context.TODO(), cacheKey, data, 5*time.Minute)

    return posts, nil
}
```

## Troubleshooting

### Common Issues

#### 1. Low Recommendation Quality

**Symptom**: Recommendations seem random or poor quality

**Solutions**:
- Ensure sufficient training data (>100 interactions per user)
- Check if users have proper metadata (interests, demographics)
- Verify edge weights are appropriate (0.0-1.0 range)
- Monitor spam detection - may be filtering too aggressively

```bash
# Check user engagement metrics
curl http://localhost:8080/api/persuasiveness/user_123

# Check graph connectivity
curl http://localhost:8080/api/stats
```

#### 2. High Memory Usage

**Symptom**: Server using excessive memory

**Solutions**:
- Reduce embedding dimensions in configuration
- Implement periodic graph cleanup
- Use edge decay to remove old connections
- Monitor node/edge growth

```bash
# Check current graph size
curl http://localhost:8080/api/stats

# Apply edge decay
curl -X POST http://localhost:8080/api/decay
```

#### 3. Slow Response Times

**Symptom**: API responses taking >100ms

**Solutions**:
- Add Redis caching layer
- Reduce recommendation limits
- Use async training instead of inline
- Monitor database query performance

#### 4. Training Not Improving

**Symptom**: Recommendation quality not improving over time

**Solutions**:
- Check if training examples are being added
- Verify label quality (positive/negative examples)
- Adjust learning rate
- Monitor training batch size

```go
// Check training data
gnn := socialGNN.(*engine.GNN)
if gnn.HybridGNN != nil {
    trainingCount := len(gnn.HybridGNN.TrainingData)
    log.Printf("Training examples: %d", trainingCount)
}
```

### Performance Optimization

#### Database Indexing

```javascript
// MongoDB indexes for better performance
db.users.createIndex({ "updated_at": 1 });
db.posts.createIndex({ "author": 1, "created_at": -1 });
db.interactions.createIndex({ "user_id": 1, "created_at": -1 });
db.likes.createIndex({ "user_id": 1, "post_id": 1 }, { unique: true });
```

#### API Rate Limiting

```go
import "golang.org/x/time/rate"

func rateLimitMiddleware(limiter *rate.Limiter) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            if !limiter.Allow() {
                http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
                return
            }
            next.ServeHTTP(w, r)
        })
    }
}
```

#### Monitoring and Metrics

```go
import "github.com/prometheus/client_golang/prometheus"

var (
    recommendationDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "socialgnn_recommendation_duration_seconds",
            Help: "Duration of recommendation requests",
        },
        []string{"user_type"},
    )

    graphSize = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "socialgnn_graph_size",
            Help: "Current size of the graph",
        },
        []string{"type"},
    )
)

func init() {
    prometheus.MustRegister(recommendationDuration)
    prometheus.MustRegister(graphSize)
}
```

For more help, check the [RULES.md](RULES.md) documentation for detailed algorithm explanations.
