package engine

import (
	"math"
	"strings"
	"time"
)

// InterestGraph builds connections between users based on interests and proximity
type InterestGraph struct {
	engine *Engine
}

// NewInterestGraph creates a new interest-based graph builder
func NewInterestGraph(engine *Engine) *InterestGraph {
	return &InterestGraph{engine: engine}
}

// Interest represents a weighted interest for a user
type Interest struct {
	Name   string  `json:"name"`
	Weight float64 `json:"weight"` // 0.0 to 1.0
}

// Location represents geographic coordinates
type Location struct {
	Lat  float64 `json:"lat"`
	Lng  float64 `json:"lng"`
	Name string  `json:"name"` // e.g., "Downtown Riverside", "Rural Route 15"
}

// UserProfile contains rich user data for interest matching
type UserProfile struct {
	UserID      string     `json:"user_id"`
	Interests   []Interest `json:"interests"`
	Location    *Location  `json:"location,omitempty"`
	Age         int        `json:"age,omitempty"`
	Occupation  string     `json:"occupation,omitempty"`
	JoinedAt    time.Time  `json:"joined_at"`
	ActivityTags []string  `json:"activity_tags"` // e.g., ["early_bird", "night_owl", "weekend_active"]
}

// BuildInterestConnections creates edges between all users based on interest similarity
func (ig *InterestGraph) BuildInterestConnections(profiles []UserProfile) error {
	for i, user1 := range profiles {
		for j, user2 := range profiles {
			if i >= j { // Avoid duplicates and self-connections
				continue
			}

			// Calculate combined similarity score
			similarity := ig.calculateUserSimilarity(user1, user2)

			if similarity > 0.1 { // Only create edges above threshold
				edgeType := ig.determineEdgeType(user1, user2, similarity)

				// Create bidirectional edges
				err1 := ig.engine.AddEdge(user1.UserID, user2.UserID, similarity, edgeType)
				err2 := ig.engine.AddEdge(user2.UserID, user1.UserID, similarity, edgeType)

				if err1 != nil || err2 != nil {
					continue // Skip if nodes don't exist yet
				}
			}
		}
	}

	return nil
}

// calculateUserSimilarity computes comprehensive similarity between two users
func (ig *InterestGraph) calculateUserSimilarity(user1, user2 UserProfile) float64 {
	// Interest similarity (weighted 60%)
	interestSim := ig.calculateInterestSimilarity(user1.Interests, user2.Interests)

	// Geographic proximity (weighted 25%)
	proximitySim := ig.calculateProximitySimilarity(user1.Location, user2.Location)

	// Demographic similarity (weighted 10%)
	demoSim := ig.calculateDemographicSimilarity(user1, user2)

	// Activity pattern similarity (weighted 5%)
	activitySim := ig.calculateActivitySimilarity(user1.ActivityTags, user2.ActivityTags)

	// Weighted combination
	totalSimilarity := (interestSim * 0.6) + (proximitySim * 0.25) + (demoSim * 0.1) + (activitySim * 0.05)

	return math.Max(0.0, math.Min(1.0, totalSimilarity))
}

// calculateInterestSimilarity uses cosine similarity with weighted interests
func (ig *InterestGraph) calculateInterestSimilarity(interests1, interests2 []Interest) float64 {
	if len(interests1) == 0 || len(interests2) == 0 {
		return 0.0
	}

	// Create interest vectors
	allInterests := ig.getAllUniqueInterests(interests1, interests2)
	if len(allInterests) == 0 {
		return 0.0
	}

	vector1 := ig.createInterestVector(interests1, allInterests)
	vector2 := ig.createInterestVector(interests2, allInterests)

	// Calculate cosine similarity
	return ig.cosineSimilarity(vector1, vector2)
}

// calculateProximitySimilarity based on geographic distance (Riverside context)
func (ig *InterestGraph) calculateProximitySimilarity(loc1, loc2 *Location) float64 {
	if loc1 == nil || loc2 == nil {
		return 0.3 // Default similarity for unknown locations in small town
	}

	// Calculate distance using Haversine formula
	distance := ig.haversineDistance(loc1.Lat, loc1.Lng, loc2.Lat, loc2.Lng)

	// In Riverside (small town), convert to similarity
	// 0 miles = 1.0 similarity, 10+ miles = 0.0 similarity
	if distance <= 1.0 {
		return 1.0 // Very close neighbors
	} else if distance <= 5.0 {
		return 0.8 // Same neighborhood
	} else if distance <= 10.0 {
		return 0.4 // Same town area
	} else {
		return 0.1 // Different areas
	}
}

// calculateDemographicSimilarity based on age, occupation, etc.
func (ig *InterestGraph) calculateDemographicSimilarity(user1, user2 UserProfile) float64 {
	similarity := 0.0

	// Age similarity
	if user1.Age > 0 && user2.Age > 0 {
		ageDiff := math.Abs(float64(user1.Age - user2.Age))
		ageSim := math.Max(0, 1.0-(ageDiff/30.0)) // 30-year span
		similarity += ageSim * 0.5
	}

	// Occupation similarity
	if user1.Occupation != "" && user2.Occupation != "" {
		occSim := ig.calculateOccupationSimilarity(user1.Occupation, user2.Occupation)
		similarity += occSim * 0.5
	}

	return similarity
}

// calculateActivitySimilarity based on activity patterns
func (ig *InterestGraph) calculateActivitySimilarity(tags1, tags2 []string) float64 {
	if len(tags1) == 0 || len(tags2) == 0 {
		return 0.5 // Default similarity
	}

	matches := 0
	total := len(tags1) + len(tags2)

	for _, tag1 := range tags1 {
		for _, tag2 := range tags2 {
			if tag1 == tag2 {
				matches++
			}
		}
	}

	return float64(matches*2) / float64(total) // Jaccard-like similarity
}

// Helper functions

func (ig *InterestGraph) getAllUniqueInterests(interests1, interests2 []Interest) []string {
	interestMap := make(map[string]bool)

	for _, interest := range interests1 {
		interestMap[strings.ToLower(interest.Name)] = true
	}
	for _, interest := range interests2 {
		interestMap[strings.ToLower(interest.Name)] = true
	}

	var result []string
	for interest := range interestMap {
		result = append(result, interest)
	}

	return result
}

func (ig *InterestGraph) createInterestVector(interests []Interest, allInterests []string) []float64 {
	vector := make([]float64, len(allInterests))
	interestMap := make(map[string]float64)

	for _, interest := range interests {
		interestMap[strings.ToLower(interest.Name)] = interest.Weight
	}

	for i, interest := range allInterests {
		if weight, exists := interestMap[interest]; exists {
			vector[i] = weight
		}
	}

	return vector
}

func (ig *InterestGraph) cosineSimilarity(a, b []float64) float64 {
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

func (ig *InterestGraph) haversineDistance(lat1, lng1, lat2, lng2 float64) float64 {
	const earthRadius = 3959 // miles

	lat1Rad := lat1 * math.Pi / 180
	lng1Rad := lng1 * math.Pi / 180
	lat2Rad := lat2 * math.Pi / 180
	lng2Rad := lng2 * math.Pi / 180

	dlat := lat2Rad - lat1Rad
	dlng := lng2Rad - lng1Rad

	a := math.Sin(dlat/2)*math.Sin(dlat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*
		math.Sin(dlng/2)*math.Sin(dlng/2)

	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return earthRadius * c
}

func (ig *InterestGraph) calculateOccupationSimilarity(occ1, occ2 string) float64 {
	// Define occupation groups for small town context
	occupationGroups := map[string][]string{
		"education":    {"teacher", "principal", "tutor", "librarian"},
		"healthcare":   {"nurse", "doctor", "paramedic", "veterinarian"},
		"trades":       {"mechanic", "electrician", "plumber", "carpenter", "welder"},
		"agriculture":  {"rancher", "farmer", "ranch hand", "agricultural worker"},
		"service":      {"restaurant owner", "shop owner", "barber", "cleaner"},
		"government":   {"mayor", "sheriff", "clerk", "postal worker"},
		"technology":   {"software developer", "it support", "programmer"},
		"retail":       {"store manager", "cashier", "sales associate"},
	}

	occ1Lower := strings.ToLower(occ1)
	occ2Lower := strings.ToLower(occ2)

	// Exact match
	if occ1Lower == occ2Lower {
		return 1.0
	}

	// Group similarity
	for _, group := range occupationGroups {
		inGroup1 := false
		inGroup2 := false

		for _, occupation := range group {
			if strings.Contains(occ1Lower, occupation) || strings.Contains(occupation, occ1Lower) {
				inGroup1 = true
			}
			if strings.Contains(occ2Lower, occupation) || strings.Contains(occupation, occ2Lower) {
				inGroup2 = true
			}
		}

		if inGroup1 && inGroup2 {
			return 0.7 // High similarity for same occupation group
		}
	}

	return 0.0
}

func (ig *InterestGraph) determineEdgeType(user1, user2 UserProfile, similarity float64) string {
	if similarity >= 0.8 {
		return "strong_interest_match"
	} else if similarity >= 0.6 {
		return "interest_match"
	} else if user1.Location != nil && user2.Location != nil {
		distance := ig.haversineDistance(user1.Location.Lat, user1.Location.Lng,
			user2.Location.Lat, user2.Location.Lng)
		if distance <= 1.0 {
			return "neighbor"
		}
	}
	return "weak_connection"
}

// LoadRiversideInterestProfiles creates sample interest profiles for testing
func (ig *InterestGraph) LoadRiversideInterestProfiles() []UserProfile {
	// Riverside, WY coordinates (approximate)
	downtownLoc := &Location{Lat: 41.0369, Lng: -106.3167, Name: "Downtown Riverside"}
	ruralNorthLoc := &Location{Lat: 41.0400, Lng: -106.3200, Name: "North Rural"}
	ruralSouthLoc := &Location{Lat: 41.0330, Lng: -106.3100, Name: "South Rural"}

	profiles := []UserProfile{
		{
			UserID: "sarah",
			Interests: []Interest{
				{Name: "photography", Weight: 0.9},
				{Name: "hiking", Weight: 0.8},
				{Name: "education", Weight: 0.7},
				{Name: "nature", Weight: 0.8},
				{Name: "community_events", Weight: 0.6},
			},
			Location:     downtownLoc,
			Age:          34,
			Occupation:   "teacher",
			JoinedAt:     time.Now().Add(-2 * 365 * 24 * time.Hour),
			ActivityTags: []string{"early_bird", "weekend_active", "social"},
		},
		{
			UserID: "jake",
			Interests: []Interest{
				{Name: "cars", Weight: 1.0},
				{Name: "fishing", Weight: 0.9},
				{Name: "sports", Weight: 0.7},
				{Name: "hunting", Weight: 0.8},
				{Name: "outdoors", Weight: 0.6},
			},
			Location:     ruralNorthLoc,
			Age:          29,
			Occupation:   "mechanic",
			JoinedAt:     time.Now().Add(-1 * 365 * 24 * time.Hour),
			ActivityTags: []string{"night_owl", "weekend_active", "practical"},
		},
		{
			UserID: "emma",
			Interests: []Interest{
				{Name: "technology", Weight: 0.9},
				{Name: "cooking", Weight: 0.8},
				{Name: "travel", Weight: 0.7},
				{Name: "photography", Weight: 0.6},
				{Name: "business", Weight: 0.5},
			},
			Location:     downtownLoc,
			Age:          31,
			Occupation:   "software developer",
			JoinedAt:     time.Now().Add(-6 * 30 * 24 * time.Hour),
			ActivityTags: []string{"night_owl", "weekday_active", "creative"},
		},
		{
			UserID: "tom",
			Interests: []Interest{
				{Name: "woodworking", Weight: 1.0},
				{Name: "history", Weight: 0.8},
				{Name: "gardening", Weight: 0.9},
				{Name: "agriculture", Weight: 0.7},
				{Name: "crafts", Weight: 0.6},
			},
			Location:     ruralSouthLoc,
			Age:          65,
			Occupation:   "retired rancher",
			JoinedAt:     time.Now().Add(-3 * 365 * 24 * time.Hour),
			ActivityTags: []string{"early_bird", "daily_active", "traditional"},
		},
	}

	return profiles
}