# SocialGNN Ranking Rules and Algorithms

Comprehensive documentation of all ranking rules, algorithms, and decision-making logic in the SocialGNN system.

## Table of Contents

- [Overview](#overview)
- [Core Architecture](#core-architecture)
- [User Authority System](#user-authority-system)
- [Spam Detection](#spam-detection)
- [Followback Health System](#followback-health-system)
- [Follow Harvesting Detection](#follow-harvesting-detection)
- [Engagement Scoring](#engagement-scoring)
- [Content Ranking](#content-ranking)
- [Comment Ranking](#comment-ranking)
- [Neural Network Learning](#neural-network-learning)
- [Temporal Decay](#temporal-decay)

## Overview

SocialGNN uses a hybrid approach combining:

1. **Neural Network Learning** (60%): Learned user preferences from interactions
2. **Rule-Based Heuristics** (40%): Hand-crafted social signals and behavioral rules

This ensures both intelligent learning and robust fallback behavior with explainable decisions.

## Core Architecture

### Graph Structure

```
Users ←→ Users     (friend/follow relationships)
  ↓       ↓
Users → Posts      (authorship)
  ↓       ↓
Users → Posts      (likes, shares, bookmarks)
  ↓       ↓
Users → Comments   (authorship)
  ↓       ↓
Comments → Posts   (parent relationship)
```

### Node Types and Properties

#### UserNode
- **Core Metrics**: Persuasiveness, spam probability, followback health
- **Authority Levels**: Admin (4x multiplier) > Human (1x) > Agent (0.8x)
- **Social Health**: Based on followback ratios and engagement patterns

#### PostNode
- **Engagement Metrics**: Views, likes, shares, comments, bookmarks
- **Quality Signals**: Author credibility, content resonance, spam detection
- **Temporal Factors**: Recency, sustained engagement over time

#### CommentNode
- **Ranking Factors**: Time decay, engagement, author credibility
- **Propagation**: Comment engagement flows to parent posts (30%)

## User Authority System

### Authority Hierarchy

```
Admin Users:     4.0x persuasiveness multiplier
Human Users:     1.0x baseline multiplier
Agent Users:     0.8x persuasiveness penalty
```

### Implementation (`graph.go:648-668`)

```go
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

    return finalScore
}
```

### Authority Detection Rules

**Admin Detection**:
- User metadata contains `"role": "admin"`
- Applied to: Persuasiveness scores, content recommendations, comment rankings
- Effect: Content from admins gets 4x weight boost

**Agent Detection**:
- User metadata contains `"user_type": "agent"`
- Applied to: All content and user ranking algorithms
- Effect: Agent content gets 20% penalty to promote human-generated content

## Spam Detection

### Multi-Layer Spam Detection System

#### Layer 1: Post Volume Analysis
```go
// High post volume with few friends = suspicious
if postCount >= 5 && friendCount < 2 {
    spamProbability += 0.4
}

// Very high post volume = more suspicious
if postCount >= 10 {
    spamProbability += 0.3
}
```

#### Layer 2: Engagement Quality Analysis
```go
// Low average engagement per post = content not resonating
if avgPostEngagement < 2.0 && postCount >= 3 {
    spamProbability += 0.3
}
```

#### Layer 3: Social Connection Analysis
```go
// Friend-to-post ratio too low = not building community connections
if friendToPostRatio < 0.1 && postCount >= 5 {
    spamProbability += 0.2
}
```

#### Layer 4: Follow Harvesting Detection (NEW)
```go
// Heavy penalty for follow harvesting behavior
spamProbability += followHarvestingPenalty // Up to 0.95 additional penalty
```

### Spam Thresholds

- **Spam Probability > 0.6**: User flagged as likely spammer
- **Spam Probability > 0.8**: Content heavily downranked
- **Spam Probability > 0.9**: User recommendations filtered out

### Impact on Rankings

**Content Penalties**:
```go
if authorNode.Engagement.UserSpamFlags.IsLikelySpammer {
    spamPenalty = 1.0 - (spamProbability * 0.7) // Up to 70% penalty
    finalScore *= spamPenalty
}
```

**Persuasiveness Penalties**:
```go
if userSpamFlags.IsLikelySpammer {
    spamPenalty = rawScore * userSpamFlags.SpamProbability * 0.8 // Up to 80% reduction
    persuasivenessScore = max(0.0, rawScore - spamPenalty)
}
```

## Followback Health System

### Followback Ratio Calculation

```go
followbackRate = followbacks / following  // Percentage of mutual follows
```

Where:
- `following`: Users this user follows
- `followbacks`: Number of bidirectional relationships
- `followbackRate`: Ratio from 0.0 to 1.0

### Health Score Formula (`graph.go:538-557`)

```go
func (g *Graph) calculateFollowbackHealthScore(followbackRate float64) float64 {
    // Optimal range: 60-70% followback rate
    if followbackRate >= 0.6 && followbackRate <= 0.7 {
        return 1.5  // Maximum boost in optimal range
    }

    // Below 60%: gradual increase from 1.0 to 1.5
    if followbackRate < 0.6 {
        return 1.0 + (followbackRate / 0.6 * 0.5)  // Linear increase
    }

    // Above 70%: maintain 1.5x boost (no penalty)
    return 1.5
}
```

### Health Score Impact

**Persuasiveness Boost**:
```go
if engagement.FollowbackMetrics != nil {
    persuasivenessScore *= engagement.FollowbackMetrics.HealthScore
}
```

**User Recommendations**:
```go
if candidate.Type == UserNode {
    baseUserScore := (persuasivenessBoost * 0.5) + (embeddingSimilarity * 0.3) + (engagementBoost * 0.2)
    finalScore = baseUserScore * followbackHealthBoost
}
```

### Rationale

- **60-70% optimal**: Indicates healthy social reciprocity
- **Below 60%**: Users may be overly selective or antisocial
- **Above 70%**: Still healthy, no penalty for being very reciprocal
- **No penalty above optimal**: Encourages positive social behavior

## Follow Harvesting Detection

### Detection Criteria

**Primary Trigger**: `followbackRate < 0.20 AND followerCount > 100`

This targets users who:
- Follow many users but don't follow back (low reciprocity)
- Have accumulated large follower counts (successful harvesting)

### Penalty Structure (`graph.go:561-587`)

```go
func (g *Graph) calculateFollowHarvestingPenalty(followbackRate float64, followerCount int) float64 {
    if followbackRate < 0.20 && followerCount > 100 {
        followHarvestingPenalty := 0.8  // Base 80% penalty

        // Additional penalty for extremely low followback rates
        if followbackRate < 0.10 {
            followHarvestingPenalty = 0.95  // 95% penalty for <10% followback
        } else if followbackRate < 0.15 {
            followHarvestingPenalty = 0.90  // 90% penalty for <15% followback
        }

        // Scale up penalty for massive follower counts (industrial harvesting)
        if followerCount > 500 {
            followHarvestingPenalty = min(1.0, followHarvestingPenalty + 0.1)
        } else if followerCount > 1000 {
            followHarvestingPenalty = 1.0  // Maximum penalty for massive harvesting
        }

        return followHarvestingPenalty
    }

    return 0.0  // No penalty for healthy users
}
```

### Penalty Escalation

| Followback Rate | Follower Count | Penalty | Effect |
|----------------|----------------|---------|---------|
| 15-20% | 100-500 | 80% | Heavy spam probability increase |
| 10-15% | 100-500 | 90% | Severe spam probability increase |
| <10% | 100-500 | 95% | Near-maximum spam probability |
| Any | >500 | +10% | Industrial scale penalty boost |
| Any | >1000 | 100% | Maximum penalty (flagged as spammer) |

### Integration with Spam Detection

```go
// Follow harvesting penalty directly added to spam probability
spamProbability += followHarvestingPenalty

// This can push spam probability to 1.0 for severe cases
spamMetrics.SpamProbability = min(1.0, spamProbability)
spamMetrics.IsLikelySpammer = spamProbability > 0.6
```

## Engagement Scoring

### Emotional Reactions System

```go
type EmotionalReactions struct {
    Like    int `json:"like"`    // x1 multiplier - Basic positive signal
    Love    int `json:"love"`    // x2 multiplier - Strong positive signal
    Laugh   int `json:"laugh"`   // x1 multiplier - Positive engagement
    Angry   int `json:"angry"`   // -1 multiplier - Negative signal
    Sad     int `json:"sad"`     // -0.5 multiplier - Mild negative signal
    Insight int `json:"insight"` // x5 multiplier - Highest value (knowledge sharing)
}
```

### Engagement Score Calculation (`graph.go:286-311`)

```go
func (g *Graph) calculateEngagementScore(e *EngagementMetrics) float64 {
    // Base engagement from reactions (weighted by emotional value)
    reactionScore := float64(e.Reactions.Like)*1.0 +
                    float64(e.Reactions.Love)*2.0 +
                    float64(e.Reactions.Laugh)*1.0 +
                    float64(e.Reactions.Angry)*(-1.0) +
                    float64(e.Reactions.Sad)*(-0.5) +
                    float64(e.Reactions.Insight)*5.0

    // High-value engagement actions
    actionScore := float64(e.Shares)*3.0 +        // Shares = strong endorsement
                  float64(e.Bookmarks)*2.0 +      // Bookmarks = saving for later
                  float64(e.Comments)*2.5 +       // Comments = active participation
                  float64(e.Clickthroughs)*1.0    // Clickthroughs = interest

    // Passive engagement (lower weight)
    passiveScore := float64(e.Views) * 0.1        // Views = basic exposure

    totalScore := reactionScore + actionScore + passiveScore

    // Logarithmic dampening to prevent runaway scores
    if totalScore > 0 {
        totalScore = math.Log(1+totalScore) * 5.0  // Scale for meaningful range
    }

    return math.Max(0.0, totalScore)
}
```

### Engagement Weighting Philosophy

1. **Insight Reactions** (5x): Most valuable - indicates knowledge sharing and educational content
2. **Shares** (3x): Strong endorsement - user willing to associate their identity with content
3. **Comments** (2.5x): Active engagement - requires time and thought investment
4. **Love Reactions** (2x): Stronger than basic likes
5. **Bookmarks** (2x): Intent to revisit - high personal value
6. **Likes/Laughs** (1x): Basic positive engagement
7. **Clickthroughs** (1x): Interest but minimal commitment
8. **Views** (0.1x): Passive exposure only
9. **Sad** (-0.5x): Mild negative signal
10. **Angry** (-1x): Negative signal

## Content Ranking

### Post Recommendation Algorithm (`gnn.go:618-685`)

#### Phase 1: Embedding Similarity
```go
userEmbedding, _ := gnn.MessagePassing(userID, 2)
candidateEmbedding, _ := gnn.MessagePassing(candidate.ID, 1)
embeddingSimilarity := gnn.CosineSimilarity(userEmbedding, candidateEmbedding)
```

#### Phase 2: Engagement Boost
```go
engagementBoost := 0.0
if candidate.Engagement != nil && candidate.Engagement.Score > 0 {
    // Normalize engagement score to 0-1 range using sigmoid
    engagementBoost = 1.0 / (1.0 + exp(-candidate.Engagement.Score/10.0))
}
```

#### Phase 3: Author Credibility
```go
persuasivenessBoost := 0.0
if candidate.Type == UserNode && candidate.Engagement.Persuasiveness > 0 {
    // Normalize persuasiveness to 0-1 range using sigmoid
    persuasivenessBoost = 1.0 / (1.0 + exp(-candidate.Engagement.Persuasiveness/5.0))
}
```

#### Phase 4: Content Type Weighting
```go
if candidate.Type == UserNode {
    // User ranking: 50% persuasiveness + 30% embedding similarity + 20% engagement
    baseUserScore := (persuasivenessBoost * 0.5) + (embeddingSimilarity * 0.3) + (engagementBoost * 0.2)
    finalScore = baseUserScore * followbackHealthBoost
} else {
    // Non-user ranking: 70% embedding similarity + 30% engagement
    finalScore = (embeddingSimilarity * 0.7) + (engagementBoost * 0.3)
}
```

#### Phase 5: Author Authority Weighting
```go
// Apply human vs agent content weighting for non-user content
if candidate.Type != UserNode {
    finalScore = gnn.applyContentAuthorWeighting(candidate, finalScore)
}
```

### Content Author Weighting (`graph.go:680-695`)

```go
func (g *Graph) applyContentAuthorWeighting(authorID string, baseScore float64) float64 {
    authorNode, exists := g.nodes[authorID]
    if !exists || authorNode.Type != UserNode {
        return baseScore
    }

    // Apply user role multipliers to content scoring
    return g.applyUserRoleMultipliers(authorID, baseScore)
}
```

**Effect**: Content from admin users gets 4x boost, agent content gets 0.8x penalty.

## Comment Ranking

### Comment Ranking Algorithm (`graph.go:850-932`)

Comments are ranked using a sophisticated multi-factor system:

#### Phase 1: Time Scoring
```go
age := time.Since(comment.CreatedAt).Hours()
timeScore := math.Max(0.0, 1.0 - age/168.0)  // Decay over 1 week, min 0.0
```

#### Phase 2: Engagement Scoring
```go
if comment.Engagement != nil {
    engageScore = math.Min(1.0, comment.Engagement.Score/10.0)  // Normalize to 0-1
}
```

#### Phase 3: Author Credibility
```go
authorCredibility := 0.0
if authorID := comment.Metadata["author"]; authorID != "" {
    if authorNode, exists := g.nodes[authorID]; exists && authorNode.Type == UserNode {
        if authorNode.Engagement != nil {
            // Normalize persuasiveness to 0-1 range
            authorCredibility = math.Min(1.0, authorNode.Engagement.Persuasiveness/15.0)
        }
    }
}
```

#### Phase 4: Spam Penalty Detection
```go
isSpamPenalized := false
spamPenalty := 1.0  // Default no penalty

if authorNode.Engagement.UserSpamFlags != nil && authorNode.Engagement.UserSpamFlags.IsLikelySpammer {
    spamPenalty = 1.0 - (authorNode.Engagement.UserSpamFlags.SpamProbability * 0.7)  // Up to 70% penalty
    isSpamPenalized = true
}
```

#### Phase 5: Final Ranking Score
```go
// Enhanced ranking: 30% time + 40% engagement + 30% author credibility
baseScore := (timeScore * 0.3) + (engageScore * 0.4) + (authorCredibility * 0.3)
rankScore := baseScore * spamPenalty  // Apply spam penalty
```

#### Phase 6: Promotion/Demotion Flags
```go
isPromoted := authorCredibility > 0.8 && engageScore > 0.7  // High credibility + engagement
// isSpamPenalized already calculated above
```

### Comment-to-Post Engagement Propagation

When comments receive engagement, 30% flows to the parent post:

```go
func (g *Graph) propagateCommentEngagementToPost(commentID string, engagementType string, delta int) {
    comment, exists := g.nodes[commentID]
    if !exists || comment.Type != CommentNode {
        return
    }

    parentID, exists := comment.Metadata["parent_id"]
    if !exists {
        return
    }

    // Propagate 30% of comment engagement to parent post
    propagationFactor := 0.3
    propagatedDelta := int(float64(delta) * propagationFactor)

    if propagatedDelta > 0 {
        g.UpdateEngagement(parentID.(string), engagementType, propagatedDelta)
    }
}
```

**Rationale**: Active comments indicate post quality and should boost the original content's visibility.

## Neural Network Learning

### Hybrid Architecture (`gnn.go:29-53`)

The system combines traditional GNN message passing with rule-based heuristics:

```go
type HybridGNN struct {
    Graph          *Graph
    EmbeddingDim   int             // 128-dimensional learned embeddings
    HeuristicDim   int             // 8-dimensional hand-crafted features
    Layers         []*GNNLayer     // Neural network layers
    AggregationType AggregationType
    LearningRate   float64
    TrainingData   []*TrainingExample
}
```

### Heuristic Feature Extraction (`gnn.go:249-310`)

8-dimensional feature vector combining:

1. **Persuasiveness** (normalized): `min(1.0, persuasiveness/10.0)`
2. **Engagement Score** (normalized): `min(1.0, score/20.0)`
3. **Spam Probability**: `spamFlags.SpamProbability`
4. **Followback Health**: `followbackMetrics.HealthScore / 1.5`
5. **Follow Harvesting Penalty**: `spamFlags.FollowHarvesting`
6. **Node Type Encoding**: Users=1.0, Posts=0.5, Comments=0.25
7. **Engagement Recency**: `max(0.0, 1.0 - hoursSinceUpdate/168.0)`
8. **Graph Connectivity**: `min(1.0, degree/50.0)`

### Neural Network Architecture

**Layer 1**: (128 + 8) input → 64 hidden units
**Layer 2**: 64 hidden → 128 output embedding
**Activation**: ReLU between layers, linear output

```go
// Forward pass combines embeddings with heuristics
combinedInput := append(nodeEmbedding, heuristicFeatures...)
learnedEmbedding := neuralNetwork.ForwardPass(combinedInput)
```

### Training System (`gnn.go:428-489`)

#### Training Example Collection
```go
type TrainingExample struct {
    UserID       string  `json:"user_id"`      // Source user
    TargetID     string  `json:"target_id"`    // Target content/user
    Interaction  string  `json:"interaction"`  // "like", "follow", "share", etc.
    Label        float64 `json:"label"`        // 1.0 for positive, 0.0 for negative
    Timestamp    int64   `json:"timestamp"`    // When interaction occurred
}
```

#### Mini-batch Training
```go
// Sample random batch from recent training data
batch := sampleRandomBatch(trainingData, batchSize)

for _, example := range batch {
    // Get current embeddings
    userEmbedding := hybridMessagePassing(example.UserID)
    targetEmbedding := hybridMessagePassing(example.TargetID)

    // Compute prediction and loss
    predicted := cosineSimilarity(userEmbedding, targetEmbedding)
    loss := (predicted - example.Label)²

    // Apply gradient descent
    gradient := 2.0 * (predicted - example.Label) * learningRate
    updateEmbeddings(example.UserID, example.TargetID, gradient)
}
```

### Hybrid Message Passing (`gnn.go:344-415`)

The core algorithm combining learned and heuristic features:

1. **Get Base Embedding**: Start with node's current embedding
2. **Extract Heuristics**: Generate 8D feature vector
3. **Neural Processing**: Pass combined (128+8)D vector through network
4. **Neighbor Aggregation**: Traditional message passing on neighbors
5. **Final Combination**: 70% learned + 30% neighbor aggregation

```go
func (h *HybridGNN) HybridMessagePassing(nodeID string, maxHops int) ([]float64, error) {
    // 1. Get base embedding
    baseEmbedding := getNodeEmbedding(nodeID)

    // 2. Extract heuristic features
    heuristicFeatures := h.extractHeuristicFeatures(nodeID)

    // 3. Combine and process through neural network
    combinedInput := append(baseEmbedding, heuristicFeatures...)
    learnedEmbedding := h.forwardPass(combinedInput)

    // 4. Aggregate neighbors
    neighborEmbeddings := getNeighborEmbeddings(nodeID, maxHops-1)
    aggregated := h.aggregateEmbeddings(neighborEmbeddings, weights)

    // 5. Final combination
    for i := range learnedEmbedding {
        learnedEmbedding[i] = 0.7*learnedEmbedding[i] + 0.3*aggregated[i]
    }

    return learnedEmbedding, nil
}
```

## Temporal Decay

### Edge Decay System (`graph.go:214-244`)

Connections lose strength over time to maintain relevance:

```go
func (g *Graph) ApplyDecay(decayRate float64, maxAge time.Duration) {
    now := time.Now()

    for fromNode, edges := range g.edges {
        for toNode, edge := range edges {
            age := now.Sub(edge.UpdatedAt)

            if age > maxAge {
                // Remove very old edges
                delete(edges, toNode)
                continue
            }

            // Apply exponential decay
            ageFactor := age.Hours() / maxAge.Hours()  // 0.0 to 1.0
            decayFactor := math.Exp(-decayRate * ageFactor)

            edge.Weight *= decayFactor

            // Remove edges that have decayed below threshold
            if edge.Weight < 0.1 {
                delete(edges, toNode)
            }
        }
    }
}
```

### Decay Parameters

**Default Configuration**:
- **Decay Rate**: 0.1 (moderate decay)
- **Max Age**: 168 hours (1 week)
- **Minimum Weight**: 0.1 (removal threshold)

**Decay Formula**: `newWeight = oldWeight * e^(-decayRate * ageFactor)`

**Rationale**: Recent interactions are more predictive of current preferences than old ones.

## Summary

The SocialGNN ranking system provides:

1. **Explainable Rankings**: Every score can be traced to specific factors
2. **Robust Spam Protection**: Multi-layer detection with severe penalties
3. **Social Health Optimization**: Rewards healthy community behavior
4. **Authority Recognition**: Proper weighting of user credibility levels
5. **Learning Capabilities**: Continuous improvement from user interactions
6. **Temporal Relevance**: Automatic decay of old information

The hybrid approach ensures both intelligent learning and reliable fallback behavior, making it suitable for production social media applications.